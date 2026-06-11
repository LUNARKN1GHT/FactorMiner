"""RL 因子挖掘·环境：gym 式 MDP，把语法层包成 reset()/step()，END 时算 IC 当奖励。

奖励 = 同 regime 训练段上的 |mean rank IC|（regime 断点教训：训练段必须 2021+）。
观测 = 已生成 token 的 index 序列；legal_mask() 给策略屏蔽非法动作。
本文件不依赖 torch；reward 复用现成 evaluate + calc_ic_series。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import calc_ic_series
from src.gp.evaluator import evaluate
from src.rl.tokens import Token, build_vocab, is_legal_next, rpn_to_node

MIN_DISTINCT = 4
"""退化守卫：每日截面去重中位数下限"""


class FactorEnv:
    """逐 token 生成因子的 MDP 环境"""

    def __init__(
        self,
        wide: dict[str, pd.DataFrame],
        fwd: pd.DataFrame,
        train_lo: pd.Timestamp,
        train_hi: pd.Timestamp,
        *,
        method: str = "rank",
        terminals: list[str] | None = None,
        max_len: int = 20,
        min_len: int = 2,
    ) -> None:
        self.wide = wide
        # 奖励只在同 regime 训练段上计算
        self.fwd_train = fwd.loc[(fwd.index >= train_lo) & (fwd.index < train_hi)]
        self.method = method
        self.vocab: list[Token] = build_vocab(terminals=terminals)
        self.max_len = max_len
        self.min_len = min_len
        self.reset()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def reset(self) -> list[int]:
        self.tokens: list[Token] = []
        self.seq: list[int] = []
        self.stack_depth = 0
        return list(self.seq)

    def legal_mask(self) -> np.ndarray:
        """长度=vocab_size 的 bool 数组：当前哪些动作合法（含预算收口约束）。"""
        d, n = self.stack_depth, len(self.tokens)
        remaining = self.max_len - n  # 还能放几个 token
        mask = np.zeros(self.vocab_size, dtype=bool)
        for i, token in enumerate(self.vocab):
            if not is_legal_next(token=token, stack_depth=d, length=n, min_len=self.min_len):
                continue
            if token.kind == "end":
                mask[i] = True
                continue
            # 放完之后的新栈深，及之后收口需要的步数
            new_d = (d + 1) if token.kind == "terminal" else (d - token.arity + 1)
            if remaining - 1 >= new_d:
                mask[i] = True
        if not mask.any() and d == 1:
            # 如果实在没别的就直接终止
            mask[self.vocab_size - 1] = True
        return mask

    def step(self, action: int) -> tuple[list[int], float, bool, dict]:
        token = self.vocab[action]
        self.tokens.append(token)
        self.seq.append(action)
        if token.kind == "terminal":
            self.stack_depth += 1
        elif token.kind == "op":
            self.stack_depth += 1 - token.arity

        done = token.kind == "end" or len(self.tokens) >= self.max_len
        reward: float = 0.0
        info: dict = {}
        if done:
            reward, info = self._reward()
        return list(self.seq), reward, done, info

    def _reward(self) -> tuple[float, dict]:
        """END 时：建树 → 求值 → 同 regime 训练段 |mean IC|。非法/退化 → 0。"""
        try:
            tree = rpn_to_node(self.tokens)
        except ValueError:
            return 0.0, {"expr": None, "reason": "invalid_rpn"}
        try:
            fac = evaluate(tree, self.wide)
        except Exception:
            return 0.0, {"expr": str(tree), "reason": "eval_error"}
        if fac.nunique(axis=1).median() < MIN_DISTINCT:  # 退化守卫
            return 0.0, {"expr": str(tree), "reason": "degenerate"}
        ic = calc_ic_series(fac, self.fwd_train, method=self.method).dropna()
        if len(ic) < 20:
            return 0.0, {"expr": str(tree), "reason": "too_few_ic"}
        return float(abs(ic.mean())), {"expr": str(tree), "ic": float(ic.mean())}


if __name__ == "__main__":
    # 自测：造小批假数据（无需真实行情/torch），跑随机 rollout，验状态机+mask
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=80)
    stocks = [f"s{i}" for i in range(30)]
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = {
        t: pd.DataFrame(rng.lognormal(size=(80, 30)), index=dates, columns=stocks)
        for t in terminals
    }
    fwd = pd.DataFrame(rng.normal(0, 0.02, size=(80, 30)), index=dates, columns=stocks)

    env = FactorEnv(wide, fwd, dates[0], dates[-1] + pd.Timedelta(days=1), terminals=terminals)
    print(f"词表大小 = {env.vocab_size}")

    for ep in range(8):
        env.reset()
        done = False
        steps = 0
        while not done:
            legal = np.flatnonzero(env.legal_mask())
            assert legal.size > 0, "出现无合法动作的死局！"  # mask 必须永远有出路
            a = int(rng.choice(legal))
            _, r, done, info = env.step(a)
            steps += 1
        print(f"ep{ep}: 长度={steps:2d} 奖励|IC|={r:.4f}  {info.get('expr')}")

    print("✅ 环境自测通过：随机 rollout 全程合法、能收口、有奖励")
