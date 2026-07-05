"""RL 因子挖掘·QuantFactor REINFORCE 环境：因子池组合奖励 + IR 时变阈值塑形。

复用 `FactorEnv` 的 token 语法 / legal_mask / stepping，只重载 `_reward()`：候选因子
求值 → 退化守卫 → 加入跨 episode 持久的 `AlphaPool` → 按论文 (arXiv:2409.05144)
式 14 用时变阈值对组合 IR 塑形：

    reward = combined_ic − λ·1{combined_ir ≤ clip[(t−α)·η, 0, δ]}

训练早期 (t < α) 阈值钳在 0（几乎不罚低 IR 因子，鼓励探索）；训练后期阈值爬升到
δ，逼策略往训练后段找更稳的因子。t 是训练循环里的 episode/global step，由外层每个
episode 开始前赋给 `self.global_step`（不是环境内部的 token 步数）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.evaluator import evaluate
from src.core.tree import Node
from src.rl.alpha_pool import AlphaPool
from src.rl.env import MIN_DISTINCT, FactorEnv
from src.rl.tokens import rpn_to_node


def _to_expr(node: Node) -> str:
    """渲染成带时序窗口、可读的表达式串，如 ``ts_mean(close,5)``。

    不能用 ``str(node)``——`Node.__str__` 不显示时序窗口（`node.value`），会把不同
    窗口的因子（如 ``ts_mean(close,5)`` 和 ``ts_mean(close,20)``）误判成同一个
    dedup key（`src/llm/clean.py::to_expr` 已踩过这个坑，此处不跨生成器 import，
    按项目"生成器互不依赖"的约定就地写一份等价逻辑）。
    """
    if not node.children:
        return node.name  # type: ignore[return-value]
    args = [_to_expr(c) for c in node.children]
    if node.value is not None:
        args.append(str(node.value))
    return f"{node.name}({','.join(args)})"


class PoolFactorEnv(FactorEnv):
    """QuantFactor REINFORCE 环境：奖励看因子池组合表现，不看单因子自身 IC。"""

    def __init__(
        self,
        wide: dict[str, pd.DataFrame],
        fwd: pd.DataFrame,
        train_lo: pd.Timestamp,
        train_hi: pd.Timestamp,
        pool: AlphaPool,
        *,
        method: str = "rank",
        terminals: list[str] | None = None,
        max_len: int = 20,
        min_len: int = 2,
        ir_alpha: float = 180_000.0,
        ir_eta: float = 1e-6,
        ir_delta: float = 0.3,
        ir_lambda: float = 0.02,
    ) -> None:
        """初始化环境（参数含义同 `FactorEnv`，另加因子池与 IR 塑形超参）。

        Args:
            pool (AlphaPool): 跨 episode 持久的因子池，由训练循环创建并持有，
                **不**在 `reset()` 里清空（`reset()` 只清 token 生成状态）。
                构造 `pool` 时传入的 `fwd` 应与本环境的训练段一致。
            ir_alpha (float): IR 阈值时变调度的步数偏移（论文默认 180000）。
            ir_eta (float): 阈值爬升速率（论文默认 1/1e6）。
            ir_delta (float): 阈值上限（论文默认 0.3）。
            ir_lambda (float): 触发惩罚的幅度（论文默认 0.02）。
        """
        super().__init__(
            wide=wide,
            fwd=fwd,
            train_lo=train_lo,
            train_hi=train_hi,
            method=method,
            terminals=terminals,
            max_len=max_len,
            min_len=min_len,
            parsimony=0.0,  # 本环境靠 IR 阈值塑形而非简约惩罚控质量，论文原设计没有 parsimony 项
        )
        self.pool = pool
        self.ir_alpha = ir_alpha
        self.ir_eta = ir_eta
        self.ir_delta = ir_delta
        self.ir_lambda = ir_lambda
        self.global_step = 0
        """训练循环每个 episode 开始前赋值，驱动 IR 阈值随训练进度爬升。"""

    def _reward(self) -> tuple[float, dict]:
        """建树 → 求值 → 入池拟合 → 按 IR 时变阈值塑形奖励。

        Returns:
            tuple[float, dict]: reward 与 info（含 expr/ic/ir/threshold/size，
                或失败时的 reason），失败情形同 `FactorEnv._reward`：
                RPN 非法 / 求值报错 / 截面退化。
        """
        try:
            tree = rpn_to_node(self.tokens)
        except ValueError:
            return 0.0, {"expr": None, "reason": "invalid_rpn"}
        try:
            fac = evaluate(tree, self.wide)
        except Exception:
            return 0.0, {"expr": _to_expr(tree), "reason": "eval_error"}
        if fac.nunique(axis=1).median() < MIN_DISTINCT:
            return 0.0, {"expr": _to_expr(tree), "reason": "degenerate"}

        expr = _to_expr(tree)
        combined_ic, combined_ir = self.pool.add_candidate(expr=expr, tree=tree, factor=fac)
        threshold = float(np.clip((self.global_step - self.ir_alpha) * self.ir_eta, 0.0, self.ir_delta))
        penalty = self.ir_lambda if combined_ir <= threshold else 0.0
        reward = combined_ic - penalty
        return float(reward), {
            "expr": expr,
            "ic": combined_ic,
            "ir": combined_ir,
            "threshold": threshold,
            "size": tree.size(),
        }


if __name__ == "__main__":
    # 自测：假数据验证「跨 episode 持久池子 + IR 阈值随 global_step 爬升」
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=120)
    stocks = [f"s{i}" for i in range(30)]
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = {
        t: pd.DataFrame(rng.lognormal(size=(120, 30)), index=dates, columns=stocks)
        for t in terminals
    }
    fwd = pd.DataFrame(rng.normal(0, 0.02, size=(120, 30)), index=dates, columns=stocks)

    pool = AlphaPool(fwd=fwd, capacity=5, gd_steps=30, seed=0)
    env = PoolFactorEnv(
        wide, fwd, dates[0], dates[-1] + pd.Timedelta(days=1), pool, terminals=terminals, max_len=8
    )

    for ep in range(6):
        env.global_step = ep * 50_000  # 模拟训练推进，阈值应逐步抬升
        env.reset()
        done = False
        while not done:
            legal = np.flatnonzero(env.legal_mask())
            assert legal.size > 0, "出现无合法动作的死局！"
            a = int(rng.choice(legal))
            _, r, done, info = env.step(a)
        print(
            f"ep{ep} step={env.global_step:>7d} 阈值={info.get('threshold', 0):.3f} "
            f"reward={r:.4f} 池大小={len(pool.members)}  {info.get('expr')}"
        )

    print("✅ PoolFactorEnv 自测通过：跨 episode 池子持久、IR 阈值随 global_step 爬升")
