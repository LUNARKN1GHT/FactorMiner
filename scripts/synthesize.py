"""多因子合成：rank 归一 + 符号对齐 + 等权平均成复合因子，评估 IC/显著性/多空净值。

候选 = 每折 train|ICIR| 选中的因子（跨折集成）。复合因子已符号对齐为正向预测，
回测多头直接买顶分位、无需翻转。
"""

import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src.backtest.simple import run_backtest
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.gp.evaluator import evaluate, to_wide


def newey_west_tstat(ic: pd.Series, horizon: int) -> float:
    """重叠矫正后的 t 值（Bartlett 核，lag=horizon-1）。"""
    x = ic.to_numpy()
    n = len(x)
    mu = float(x.mean())
    dev = x - mu
    var = float(dev @ dev) / n
    for k in range(1, horizon):
        gamma_k = float(dev[k:] @ dev[:-k]) / n
        var += 2 * (1 - k / horizon) * gamma_k
    se = np.sqrt(var / n)
    return mu / se if se > 0 else 0.0


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

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    close = to_panel(prices, "close")
    open_p = to_panel(prices, "open")
    exec_ret = open_p.shift(-(1 + hold)) / open_p.shift(-1) - 1  # T+1 开盘成交

    # ---- 候选池：每折选中因子（去重）----
    candidates = []  # (expr, 对齐后的 rank 矩阵)
    seen = set()
    for fold in folds:
        pick = max(fold["rows"], key=lambda r: abs(r["train_icir"]))
        if pick["expr"] in seen:
            continue
        seen.add(pick["expr"])
        fac = evaluate(pick["tree"], wide)
        r = fac.rank(axis=1, pct=True)  # 截面 rank 归一到 [0,1]
        if pick["train_icir"] < 0:  # 符号对齐：负向因子翻成正向
            r = 1 - r
        candidates.append((pick["expr"], r))

    print(f"=== 候选因子（{len(candidates)} 个，已 rank 归一 + 符号对齐）===")
    for expr, _ in candidates:
        print(f"  {expr}")

    # ---- 候选间相关性：看合成有没有去相关价值 ----
    flat = pd.DataFrame({expr: r.stack() for expr, r in candidates})
    print("\n=== 候选两两相关（越低，合成增益越大）===")
    print(flat.corr().round(2).to_string())

    # ---- 等权合成 ----
    composite: pd.DataFrame = sum(r for _, r in candidates) / len(candidates)

    # ---- 评估：单因子 vs 复合 ----
    def _stats(fac: pd.DataFrame, tag: str) -> None:
        ic = calc_ic_series(fac, fwd, method=method).dropna()
        icir = ic.mean() / ic.std()
        nw = newey_west_tstat(ic, horizon)
        print(f"  {tag:<28} ICIR={icir:+.4f}  NW_t={nw:+.2f}  IC均值={ic.mean():+.4f}")

    print("\n=== 全样本 IC 战绩：单因子 vs 复合 ===")
    for expr, r in candidates:
        _stats(r, expr[:28])
    _stats(composite, "★ 复合因子（等权）")

    # ---- 分组单调性：复合因子分 n_groups 组，各组平均未来日收益 ----
    print("\n=== 复合因子分组单调性（各组平均未来日收益，bp）===")
    ranks = composite.rank(axis=1, pct=True)
    fwd_al = fwd.reindex_like(composite)
    for g in range(n_groups):
        lo, hi = g / n_groups, (g + 1) / n_groups
        mask = (ranks > lo) & (ranks <= hi)
        grp = fwd_al.where(mask).stack().mean()  # 该股所有平均收益
        bar = "█" * int(max(0, grp * 1e4))
        print(f"  G{g + 1} [{lo:.1f}-{hi:.1f}]  {grp * 1e4:+6.1f}bp  {bar}")

    # ---- 换手处理：平滑复合因子，扫描窗口看净值、多空是否能转正 ---
    print("\n=== 平滑窗口 vs 净多空（含 10bp 手续费）===")
    for w in (1, 3, 5, 10, 20):
        sm = composite.rolling(w, min_periods=1).mean()  # 按天平滑，持仓变黏 → 降换手
        ic = calc_ic_series(sm, fwd, method=method).dropna()
        icir = ic.mean() / ic.std()
        res = run_backtest(
            factor_df=sm,
            fwd_ret=fwd,
            close=close,
            holding_period=hold,
            n_groups=n_groups,
            commission=comm,
        )["long_short"]
        print(
            f"  w={w:>2}  ICIR={icir:+.3f}  "
            f"净多空 sharpe={res['sharpe']:+.3f}  年化={res['ann_return'] * 100:+6.2f}%"
        )

    # ---- 回测：多空对冲（纯 alpha）+ 现实多头 T+1 ----
    def _bt(ret: pd.DataFrame) -> dict:
        return run_backtest(
            factor_df=composite,
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

    res_c, res_e = _bt(fwd), _bt(exec_ret)
    print("\n=== 复合因子回测 ===")
    _line("多空对冲（纯alpha）", res_c["long_short"])  # 剥离 beta，头条口径
    _line("现实多头·收盘T0", res_c["long_only"])
    _line("现实多头·T+1开盘", res_e["long_only"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/synthesize.py <walkforward.pkl 路径>")
    main(sys.argv[1])
