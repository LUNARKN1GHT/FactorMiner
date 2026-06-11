"""统一评估入口：一个 pkl 进来，数据只读一次、每折树只求值一次，
一次跑完「现实回测三层 + Newey-West 显著性 + Deflated Sharpe 存活线」。

方向修正：回测多头按 train_icir 符号翻转因子（负 ICIR=反转信号，买底分位）；
显著性/deflated 看 |NW_t|，与方向无关，用原始因子。
"""

import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.backtest.simple import run_backtest
from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series

GAMMA = 0.5772156649015329  # 欧拉–马歇罗尼常数


# ---- 显著性数学（与 significance.py / deflated.py 同源，这里自带一份保持自洽）----
def newey_west_tstat(ic: pd.Series, horizon: int) -> tuple[float, float]:
    """返回 (naive_t, nw_t)：重叠自相关矫正前后的 t 值。lag = horizon-1。"""
    x = ic.to_numpy()
    n = len(x)
    mu = float(x.mean())
    naive_se = float(x.std(ddof=1)) / np.sqrt(n)
    naive_t = mu / naive_se if naive_se > 0 else 0.0

    dev = x - mu
    var = float(dev @ dev) / n
    lag = horizon - 1
    for k in range(1, lag + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / n
        weight = 1 - k / (lag + 1)  # Bartlett 核
        var += 2 * weight * gamma_k
    nw_se = np.sqrt(var / n)
    nw_t = mu / nw_se if nw_se > 0 else 0.0
    return naive_t, nw_t


def expected_max_t(n_trials: int) -> float:
    """N 次独立零假设检验中 t 最大值的期望。"""
    a = norm.ppf(1 - 1 / n_trials)
    b = norm.ppf(1 - 1 / (n_trials * np.e))
    return (1 - GAMMA) * a + GAMMA * b


def breakeven_n(t_obs: float) -> int:
    """反解 E[max t]=|t_obs| 的 N：有效试验数 < 它，因子才算真。"""
    target = abs(t_obs)
    n = 2
    while expected_max_t(n) < target and n < 10_000_000:
        n += 1
    return n


def deflated(t_obs: float, n_trials: int) -> tuple[float, float]:
    """返回 (DSR, deflated_p)。"""
    emax = expected_max_t(n_trials)
    dsr = float(norm.cdf(abs(t_obs) - emax))
    p_single = float(2 * norm.sf(abs(t_obs)))
    p_def = 1 - (1 - p_single) ** n_trials
    return dsr, p_def


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    folds = payload["folds"]

    with open(payload["config_path"]) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    horizon = cfg["evaluation"].get("holding_period", 5)
    hold = cfg["backtest"]["holding_period"]
    n_groups = cfg["evaluation"].get("n_groups", 5)
    comm = cfg["backtest"]["commission"]

    # ---- 数据只读一次 ----
    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    close = to_panel(prices, "close")
    open_p = to_panel(prices, "open")
    exec_ret = open_p.shift(-(1 + hold)) / open_p.shift(-1) - 1  # T+1 开盘进、开盘出

    def _bt(factor_df: pd.DataFrame, ret: pd.DataFrame) -> dict:
        return run_backtest(
            factor_df=factor_df,
            fwd_ret=ret,
            close=close,
            holding_period=hold,
            n_groups=n_groups,
            commission=comm,
        )

    def _line(tag: str, m: dict) -> None:
        print(
            f"  {tag:<18} sharpe={m['sharpe']:6.3f} "
            f"年化={m['ann_return'] * 100:7.2f}% 最大回撤={m['max_drawdown'] * 100:7.2f}%"
        )

    picks_t = []  # (fold, |NW_t|)，留给最后做 N 敏感性
    for fold in folds:
        # 复刻基线选法：训练段 |ICIR| 最强者
        pick = max(fold["rows"], key=lambda r: abs(r["train_icir"]))
        tree = pick["tree"]
        signed_tr = pick["train_icir"]

        factor = evaluate(tree, wide)  # 每折只求值一次
        factor_bt = -factor if signed_tr < 0 else factor  # 方向修正，仅回测用

        # --- 显著性（用原始因子）---
        ic = calc_ic_series(factor, fwd, method=method)
        ic_te = ic.loc[(ic.index >= fold["split"]) & (ic.index < fold["test_hi"])].dropna()
        naive_t, nw_t = newey_west_tstat(ic_te, horizon)
        icir = float(ic_te.mean() / ic_te.std())
        p_nw = float(2 * norm.sf(abs(nw_t)))
        n_star = breakeven_n(nw_t)
        picks_t.append((fold["fold"], abs(nw_t)))

        # --- 回测三层（用方向修正后的因子）---
        res_c = _bt(factor_bt, fwd)
        res_e = _bt(factor_bt, exec_ret)

        flip = " [已翻转方向]" if signed_tr < 0 else ""
        print(f"\n=== fold{fold['fold']}. {pick['expr']}{flip} ===")
        print(
            f"  显著性  ICIR={icir:+.4f}  naive_t={naive_t:.2f}  NW_t={nw_t:.2f}  "
            f"p(NW)={p_nw:.4f}  {'✓' if p_nw < 0.05 else '✗'}  break-even N*={n_star}"
        )
        _line("纸面多空·收盘T0", res_c["long_short"])
        _line("现实多头·收盘T0", res_c["long_only"])
        _line("现实多头·T+1开盘", res_e["long_only"])

    # ---- 对最强折做 deflated N 敏感性 ----
    best_fold, best_t = max(picks_t, key=lambda x: x[1])
    print(
        f"\n=== fold{best_fold}（|NW_t|={best_t:.2f}）在不同有效试验数 N 下的 deflated 显著性 ==="
    )
    print(f"{'N':>6} {'E[max t]':>9} {'DSR':>7} {'deflated p':>11}")
    for n_trials in (10, 30, 100, 300, 1000):
        dsr, p_def = deflated(best_t, n_trials)
        print(f"{n_trials:>6} {expected_max_t(n_trials):>9.2f} {dsr:>7.3f} {p_def:>11.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/evaluate_run.py <walkforward.pkl 路径>")
    main(sys.argv[1])
