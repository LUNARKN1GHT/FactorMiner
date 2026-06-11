"""Part B：扣除「在一整条前沿里挑最强」的选择偏差——Deflated Sharpe，只读 pkl + 数据。

Part A 已把单因子的重叠虚高用 NW 修掉，得到 |NW_t|。
但那条因子是 GP 从成千上万个表达式里挑出来的最强者——「挑最强」=「做了 N 次检验取最大」。
这里算零假设下「N 次试验里 t 最大值的期望 E[max t]」，看实测 |NW_t| 还能不能超过它。
N 不精确可知，所以反解 break-even N*：有效试验数得小于它，因子才算真。
"""

import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series

GAMMA = 0.5772156649015329  # 欧拉–马歇罗尼常数


def newey_west_tstat(ic: pd.Series, horizon: int) -> float:
    """重叠矫正后的 t 值"""
    x = ic.to_numpy()
    n = len(x)
    mu = float(x.mean())
    dev = x - mu
    var = float(dev @ dev) / n
    lag = horizon - 1
    for k in range(1, lag + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / n
        weight = 1 - k / (lag + 1)
        var += 2 * weight * gamma_k
    nw_se = np.sqrt(var / n)
    return mu / nw_se if nw_se > 0 else 0.0


def expected_max_t(n_trails: int) -> float:
    """N 次独立零假设检验中，t 统计量最大值的期望"""
    a = norm.ppf(1 - 1 / n_trails)
    b = norm.ppf(1 - 1 / (n_trails * np.e))
    return (1 - GAMMA) * a + GAMMA * b


def breakeven_n(t_obs: float) -> int:
    """反解 E[max t] = |t_obs| 的 N：有效试验数 < 它，因子才算真。E[max t] 随 N 单调增。"""
    target = abs(t_obs)
    n = 2
    while expected_max_t(n) < target and n < 10_000_000:
        n += 1
    return n


def deflated(t_obs: float, n_trails: int) -> tuple[float, float]:
    """返回 (DSR, deflated_p)：扣除 N 次选择后，真 SR>0 的概率，与 Šidák 校正 p 值。"""
    emax = expected_max_t(n_trails)
    dsr = float(norm.cdf(abs(t_obs) - emax))
    p_single = float(2 * norm.sf(abs(t_obs)))
    p_def = 1 - (1 - p_single) ** n_trails
    return dsr, p_def


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

    # 每折选中因子的 |NW_t| 与存活线
    print("=== 每折选中因子：扣除「在 N 次试验里挑最强」后的存活线 ===")
    print(f"{'fold':>4} {'|NW_t|':>7} {'break-even N*':>14}")
    picks_t = []  # (fold, |NW_t|)，留给后面对最强折做敏感性分析
    for fold in folds:
        pick = max(fold["rows"], key=lambda r: abs(r["train_icir"]))
        ic = calc_ic_series(evaluate(pick["tree"], wide), fwd, method=method)
        ic_te = ic.loc[(ic.index >= fold["split"]) & (ic.index < fold["test_hi"])].dropna()
        t = abs(newey_west_tstat(ic_te, horizon))
        picks_t.append((fold["fold"], t))
        print(f"{fold['fold']:>4} {t:>7.2f} {breakeven_n(t):>14d}")

    # 对存活线最高的那折（唯一有希望的）做 N 敏感性分析
    best_fold, best_t = max(picks_t, key=lambda x: x[1])
    print(
        f"\n=== fold{best_fold}（|NW_t|={best_t:.2f}）在不同「有效试验数 N」下的 deflated 显著性 ==="
    )
    print(f"{'N':>6} {'E[max t]':>9} {'DSR':>7} {'deflated p':>11}")
    for n_trials in (10, 30, 100, 300, 1000):
        dsr, p_def = deflated(best_t, n_trials)
        print(f"{n_trials:>6} {expected_max_t(n_trials):>9.2f} {dsr:>7.3f} {p_def:>11.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/deflated.py <walkforward.pkl 路径>")
    main(sys.argv[1])
