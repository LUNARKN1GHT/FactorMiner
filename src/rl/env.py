"""RL 因子挖掘·环境：gym 式 MDP，把语法层包成 reset()/step()，END 时算 IC 当奖励。

奖励 = 同 regime 训练段上的 |mean rank IC|（regime 断点教训：训练段必须 2021+）。
观测 = 已生成 token 的 index 序列；legal_mask() 给策略屏蔽非法动作。
本文件不依赖 torch；reward 复用现成 evaluate + calc_ic_series。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.evaluator import evaluate
from src.evaluation.metrics import calc_ic_series
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
        """初始化因子生成环境。

        Args:
            wide (dict[str, pd.DataFrame]): 各价量字段的宽表，key 为字段名（如
                ``"close"``），value 为 shape ``(dates, stocks)`` 的 DataFrame。
            fwd (pd.DataFrame): 前向收益宽表，shape ``(dates, stocks)``，
                用于计算 IC 奖励。
            train_lo (pd.Timestamp): 训练段左端（含），奖励只在此区间计算。
            train_hi (pd.Timestamp): 训练段右端（不含），同上。
            method (str): IC 计算方式，``"rank"`` 或 ``"pearson"``。
            terminals (list[str] | None): 允许作为叶节点的字段名；``None`` 时
                由 :func:`build_vocab` 使用默认列表。
            max_len (int): 序列最大 token 数（含 END），超出则强制终止。
            min_len (int): 合法 END 之前的最少有效 token 数。
        """
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
        """词表大小（不含 BOS 符号）。"""
        return len(self.vocab)

    def reset(self) -> list[int]:
        """重置环境到初始空状态。

        Returns:
            list[int]: 空的 token 索引序列 ``[]``。
        """
        self.tokens: list[Token] = []
        self.seq: list[int] = []
        self.stack_depth = 0
        return list(self.seq)

    def legal_mask(self) -> np.ndarray:
        """返回当前时步合法动作的布尔掩码。

        在基础语法合法性之外，还施加"预算收口"约束：若放入某 token 后，
        剩余步数不足以将栈深压回 1（END 需要栈深恰好为 1），则屏蔽该 token。

        Returns:
            np.ndarray: shape ``(vocab_size,)``，dtype bool；``True`` 表示该
                token 当前可选。至少保证一个位置为 ``True``（END 兜底）。
        """
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
        """执行一步动作，更新状态机，终止时计算奖励。

        Args:
            action (int): 词表索引，必须是 :meth:`legal_mask` 中为 ``True`` 的位置。

        Returns:
            tuple[list[int], float, bool, dict]:
                - ``obs``: 当前已生成的 token 索引序列（副本）。
                - ``reward``: 终止时为 ``|mean IC|``，未终止时为 ``0.0``。
                - ``done``: 是否已到达终止状态（END token 或超长）。
                - ``info``: 附加信息，终止时含 ``"expr"``（表达式字符串）
                  及可选的 ``"ic"`` 或 ``"reason"`` 字段。
        """
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
        """计算终止时的奖励：建树 → 求值 → 同 regime 训练段 |mean IC|。

        Returns:
            tuple[float, dict]:
                - ``reward``: ``|mean rank IC|``；以下情况返回 ``0.0``：
                  RPN 非法、求值报错、截面去重中位数 < ``MIN_DISTINCT``、
                  IC 序列有效长度 < 20。
                - ``info``: 含 ``"expr"`` 与以下之一：
                  ``"ic"``（正常）或 ``"reason"``（失败原因字符串）。
        """
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
