"""给 walk-forward 选中的因子算「重叠校正后」的显著性——只读 pkl + 数据，不重跑 GP。

5 天前向收益每天滚动重叠 → IC 序列正自相关 → std(IC) 低估 → naive t = ICIR×√N 虚高。
这里对每折选中因子的测试段 IC 序列，用 Newey-West（Bartlett 核，滞后 horizon-1）把自相关补回去，
对比 naive t 与校正后 t，看砍掉重叠的谎言后还剩多少显著性。
"""

import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.gp.evaluator import evaluate, to_wide


def newey_west_tstat(ic: pd.Series, horizon: int) -> tuple[float, float]:
    """用 Newey West t检验来判断ic是否

    Args:
        ic (pd.Series): ic 序列
        horizon (int): 回看窗口长度

    Returns:
        tuple[float, float]: 同一条 IC 序列，自相关矫正前后的 t 值
    """
    x = ic.to_numpy()
    n = len(x)
    mu = float(x.mean())

    naive_se = float(x.std(ddof=1)) / np.sqrt(n)
    naive_t = mu / naive_se if naive_se > 0 else 0.0

    dev = x - mu
    var = float(dev @ dev) / n
    lag = horizon - 1
    for k in range(1, lag + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / n  # k 阶自协方差
        weight = 1 - k / (lag + 1)  # Bartlett：越远的滞后权重越小，保证方差非负
        var += 2 * weight * gamma_k
    nw_se = np.sqrt(var / n)
    nw_t = mu / nw_se if nw_se else 0.0
    return naive_t, nw_t


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    folds = payload["folds"]

    with open(payload["config_path"]) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    horizon = cfg["evaluation"].get("holding_period", 5)

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    print(
        f"{'fold':>4} {'size':>4} {'n_te':>5} {'ICIR':>9} "
        f"{'naive_t':>8} {'NW_t':>7} {'p(NW)':>8}  显著?  expr"
    )

    for fold in folds:
        # 复刻 walk-forward 的基线选法：训练段 |ICIR| 最强，质问的就是这个会被「发表」的因子
        pick = max(fold["rows"], key=lambda r: abs(r["train_icir"]))
        tree = pick["tree"]

        # 用树重算整条 IC，再切到本折测试段（pkl 只存标量 ICIR，NW 需要整条序列）
        ic = calc_ic_series(evaluate(tree, wide), fwd, method=method)
        ic_te = ic.loc[(ic.index >= fold["split"]) & (ic.index < fold["test_hi"])].dropna()

        naive_t, nw_t = newey_west_tstat(ic_te, horizon)
        icir = float(ic_te.mean() / ic_te.std())
        p_nw = float(2 * norm.sf(abs(nw_t)))  # 双侧 p 值，正态近似（N 够大时和 t 分布几乎一致）
        mark = "✓" if p_nw < 0.05 else "✗"
        print(
            f"{fold['fold']:>4} {pick['size']:>4} {len(ic_te):>5} {icir:>+9.4f} "
            f"{naive_t:>8.2f} {nw_t:>7.2f} {p_nw:>8.4f}   {mark}   {pick['expr']}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/significance.py <walkforward.pkl 路径>")
    main(sys.argv[1])
