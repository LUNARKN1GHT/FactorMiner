"""QuantFactor REINFORCE·因子池：维护固定容量的因子集合，梯度下降拟合线性组合权重。

论文 (arXiv:2409.05144) 式 13 附近描述：新因子随机初始化权重，批量梯度下降联合优化
整池权重、最小化组合值与前向收益的 MSE；池满后淘汰权重绝对值最小的因子。组合后的
IC/IR 是 `pool_env.py` 里奖励塑形（式 14）的输入。

只服务 Track 2（QuantFactor）。Track 1（AlphaGen）的池子拟合方式（官方仓库里是别的
拟合逻辑）不与此共用——两篇论文的因子池设计本就不同，混用会把"算法差异"和
"拟合方式差异"这两个变量搅在一起，对比就不干净了。

不同量纲的因子直接线性组合没有意义，所以入池前先做截面 rank 标准化（映射到
[-0.5, 0.5]），这是论文没明说但线性组合前必须做的一步，此处显式记录原因。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.tree import Node
from src.evaluation.metrics import calc_ic_series, calc_icir


@dataclass
class PoolMember:
    """因子池里的一条记录"""

    expr: str
    """因子表达式字符串，去重/记录用"""

    tree: Node
    """因子表达式树"""

    factor: pd.DataFrame
    """已做截面 rank 标准化的因子宽表，shape (date, code)"""

    weight: float
    """当前组合权重"""


class AlphaPool:
    """线性组合因子池：梯度下降拟合权重，容量满时淘汰权重最小项。"""

    def __init__(
        self,
        fwd: pd.DataFrame,
        capacity: int = 10,
        *,
        method: str = "rank",
        lr: float = 0.05,
        gd_steps: int = 200,
        seed: int = 42,
    ) -> None:
        """初始化因子池。

        Args:
            fwd (pd.DataFrame): 前向收益宽表，shape (date, code)，组合权重拟合的目标。
            capacity (int): 池子容量上限，超出后淘汰 |weight| 最小的因子。
            method (str): 组合后计算 IC 用 "rank" 还是 "pearson"。
            lr (float): 梯度下降学习率。
            gd_steps (int): 每次 `add_candidate` 重新拟合时的梯度下降步数。
            seed (int): 新因子权重随机初始化的种子。
        """
        self.fwd = fwd
        self.capacity = capacity
        self.method = method
        self.lr = lr
        self.gd_steps = gd_steps
        self.rng = np.random.default_rng(seed)
        self.members: list[PoolMember] = []

    @staticmethod
    def _standardize(factor: pd.DataFrame) -> pd.DataFrame:
        """截面 rank 标准化到 [-0.5, 0.5]，让不同量纲的因子能线性组合。"""
        return factor.rank(axis=1, pct=True) - 0.5

    def _refit(self) -> None:
        """随机初始化新加入项的权重后，批量梯度下降联合优化整池权重。

        最小化 L(w) = mean((Σ w_i z_i − y)²)（论文式 13），梯度手推为
        grad = (2/n)·Zᵀ(Zw − y)，n 为展平后的有效 (date, code) 观测数。
        """
        if not self.members:
            return
        mats = [m.factor.reindex_like(self.fwd).to_numpy() for m in self.members]
        y = self.fwd.to_numpy()
        valid = np.isfinite(y)
        for mat in mats:
            valid &= np.isfinite(mat)
        if valid.sum() == 0:
            return
        z = np.stack([mat[valid] for mat in mats], axis=1)  # (n_obs, k)
        target = y[valid]
        w = np.array([m.weight for m in self.members], dtype=float)
        n = len(target)
        for _ in range(self.gd_steps):
            resid = z @ w - target
            grad = (2.0 / n) * (z.T @ resid)
            w = w - self.lr * grad
        for m, wi in zip(self.members, w, strict=True):
            m.weight = float(wi)

    def _combined(self) -> pd.DataFrame:
        """按当前权重线性组合池内全部因子。"""
        combo = 0.0
        for m in self.members:
            combo = combo + m.weight * m.factor
        return combo

    def add_candidate(self, expr: str, tree: Node, factor: pd.DataFrame) -> tuple[float, float]:
        """候选因子入池、重新拟合、超容量淘汰，返回组合后的 (combined_ic, combined_ir)。

        永远直接把候选加进池子再拟合——RL 的奖励信号本就来自"加入这个因子后组合
        是变好还是变差"，不需要先试探再回滚。

        Args:
            expr (str): 因子表达式字符串。
            tree (Node): 因子表达式树。
            factor (pd.DataFrame): 候选因子原始（未标准化）宽表。

        Returns:
            tuple[float, float]: 入池并淘汰后的组合 IC 均值与组合 IR；
                有效 IC 天数 < 20 时返回 (0.0, 0.0)。
        """
        self.members.append(
            PoolMember(expr=expr, tree=tree, factor=self._standardize(factor), weight=0.0)
        )
        self.members[-1].weight = float(self.rng.normal(scale=0.1))
        self._refit()
        if len(self.members) > self.capacity:
            self.members.sort(key=lambda m: abs(m.weight), reverse=True)
            self.members = self.members[: self.capacity]
        ic = calc_ic_series(self._combined(), self.fwd, method=self.method).dropna()
        if len(ic) < 20:
            return 0.0, 0.0
        return float(ic.mean()), calc_icir(ic)


if __name__ == "__main__":
    # 自测：假数据验证「入池→拟合→淘汰→组合 IC/IR」全链路
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=120)
    stocks = [f"s{i}" for i in range(40)]
    fwd = pd.DataFrame(rng.normal(0, 0.02, size=(120, 40)), index=dates, columns=stocks)

    pool = AlphaPool(fwd=fwd, capacity=3, gd_steps=50, seed=0)
    for i in range(5):
        # 让因子跟 fwd 弱相关，模拟"有点信号"的候选
        noisy = fwd.rank(axis=1) + rng.normal(0, 5, size=fwd.shape)
        ic, ir = pool.add_candidate(expr=f"factor_{i}", tree=Node(name=f"f{i}", value=f"f{i}"), factor=noisy)
        print(f"候选 {i}: 组合 IC={ic:.4f} IR={ir:.4f} 池大小={len(pool.members)}")
        assert len(pool.members) <= pool.capacity, "淘汰逻辑没生效，池子超容量了"

    print("✅ AlphaPool 自测通过：入池/拟合/淘汰/组合 IC 全链路无报错")
