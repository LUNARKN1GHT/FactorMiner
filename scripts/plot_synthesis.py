"""汇报配图：复合因子的分组单调性 / 累计 IC / 换手治理曲线，存 PNG。

自包含：重建复合因子（rank 归一 + 符号对齐 + 等权），不依赖 synthesize.py 的输出。
服务器通常无中文字体，标签一律英文，避免豆腐块。
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境，必须在 import pyplot 前设
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.backtest.simple import run_backtest
from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series


def build_composite(folds: list, wide: dict) -> list:
    """复刻 synthesize 的候选 + 对齐，返回 [(expr, 对齐后的 rank 矩阵)]。"""
    out, seen = [], set()
    for fold in folds:
        pick = max(fold["rows"], key=lambda r: abs(r["train_icir"]))
        if pick["expr"] in seen:
            continue
        seen.add(pick["expr"])
        r = evaluate(pick["tree"], wide).rank(axis=1, pct=True)
        if pick["train_icir"] < 0:
            r = 1 - r
        out.append((pick["expr"], r))
    return out


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    folds = payload["folds"]

    with open(payload["config_path"]) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    hold = cfg["backtest"]["holding_period"]
    n_groups = cfg["evaluation"].get("n_groups", 5)
    comm = cfg["backtest"]["commission"]

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    close = to_panel(prices, "close")

    candidates = build_composite(folds, wide)
    composite = sum(r for _, r in candidates) / len(candidates)

    outdir = Path("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 图1：分组单调性柱状图 ----
    ranks = composite.rank(axis=1, pct=True)
    fwd_al = fwd.reindex_like(composite)
    groups, vals = [], []
    for g in range(n_groups):
        lo, hi = g / n_groups, (g + 1) / n_groups
        mask = (ranks > lo) & (ranks <= hi)
        groups.append(f"G{g + 1}")
        vals.append(fwd_al.where(mask).stack().mean() * 1e4)  # bp

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(groups, vals, color="#3b7dd8")
    ax.set_title("Composite Factor — Quantile Monotonicity")
    ax.set_ylabel("Mean forward return (bp)")
    ax.set_xlabel("Factor quantile (low → high)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "monotonicity.png", dpi=150)
    plt.close(fig)

    # ---- 图2：累计 IC 曲线 ----
    ic = calc_ic_series(composite, fwd, method=method).dropna()
    cum = ic.cumsum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cum.index, cum.to_numpy(), color="#2a9d5c", lw=1.5)
    ax.set_title(f"Composite Factor — Cumulative Rank IC  (ICIR={ic.mean() / ic.std():.3f})")
    ax.set_ylabel("Cumulative daily IC")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "cumulative_ic.png", dpi=150)
    plt.close(fig)

    # ---- 图3：换手治理曲线（平滑窗口 vs 净多空 sharpe）----
    ws, sharpes = [1, 3, 5, 10, 20], []
    for w in ws:
        sm = composite.rolling(w, min_periods=1).mean()
        res = run_backtest(
            factor_df=sm,
            fwd_ret=fwd,
            close=close,
            holding_period=hold,
            n_groups=n_groups,
            commission=comm,
        )["long_short"]
        sharpes.append(res["sharpe"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, sharpes, "o-", color="#d8703b", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Turnover Treatment — Net Long-Short Sharpe vs Smoothing")
    ax.set_ylabel("Net long-short Sharpe (10bp cost)")
    ax.set_xlabel("Smoothing window w (days)")
    ax.grid(alpha=0.3)
    for x, y in zip(ws, sharpes, strict=True):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    fig.tight_layout()
    fig.savefig(outdir / "turnover_treatment.png", dpi=150)
    plt.close(fig)

    print(f"三张图已存到 {outdir}/：monotonicity.png / cumulative_ic.png / turnover_treatment.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/plot_synthesis.py <walkforward.pkl 路径>")
    main(sys.argv[1])
