"""因子动物园·筛选器：从因子库按 test IC 取最强的几个，回测看实战。

mentor：用 IC 在测试集上挑最好的几个。
诚实提醒：测试集上选 = 选择偏差，报 winner 时仍要带 NW/deflated 的尺子
（scripts/significance.py、scripts/deflated.py）。
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.backtest.simple import run_backtest
from src.data.preprocess import to_panel
from src.gp.evaluator import evaluate, to_wide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("library", help="factor_library.pkl 路径")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bt", type=int, default=3, help="对前几名做回测")
    args = ap.parse_args()

    with open(args.library, "rb") as fh:
        lib = pickle.load(fh)
    factors: list[dict] = lib["factors"]
    split = pd.Timestamp(lib["split"])

    # ---- 按 |test IC| 排序，取 top-K（IC 可正可负，绝对值=预测强度）----
    ranked = sorted(factors, key=lambda r: abs(r["train_ic"]), reverse=True)[: args.topk]

    print(f"=== 因子库 {len(factors)} 个，按 |test IC| 取 top{args.topk} ===")
    print(f"{'#':>3} {'train_ic':>9} {'test_ic':>9} {'size':>4}  expr")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>3} {r['train_ic']:>+9.4f} {r['test_ic']:>+9.4f} {r['size']:>4}  {r['expr']}")

    # ---- 对前 args.bt 名做测试段回测 ----
    with open(lib["config_path"]) as f:
        cfg = yaml.safe_load(f)
    hold = cfg["backtest"]["holding_period"]
    n_groups = cfg["evaluation"].get("n_groups", 5)
    comm = cfg["backtest"]["commission"]

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    close = to_panel(prices, "close")
    open_p = to_panel(prices, "open")
    exec_ret = open_p.shift(-(1 + hold)) / open_p.shift(-1) - 1

    test_hi = fwd.index.max() + pd.Timedelta(days=1)

    def _slice(df: pd.DataFrame) -> pd.DataFrame:  # 只回测测试段，与 test IC 同口径
        return df.loc[(df.index >= split) & (df.index < test_hi)]

    print(f"\n=== top{args.bt} 测试段回测（含 {comm * 1e4:.0f}bp 手续费）===")
    for r in ranked[: args.bt]:
        fac = evaluate(r["tree"], wide)
        if r["test_ic"] < 0:  # 负 IC = 反转信号，多头买底分位 → 翻转
            fac = -fac
        res = run_backtest(
            factor_df=_slice(fac),
            fwd_ret=_slice(exec_ret),
            close=_slice(close),
            holding_period=hold,
            n_groups=n_groups,
            commission=comm,
        )["long_only"]
        flip = " [翻转]" if r["test_ic"] < 0 else ""
        print(
            f"  test_ic={r['test_ic']:+.4f}{flip}  多头T+1 sharpe={res['sharpe']:+.3f} "
            f"年化={res['ann_return'] * 100:+.2f}%  {r['expr']}"
        )

    tr = np.array([f["train_ic"] for f in factors])
    te = np.array([f["test_ic"] for f in factors])
    rho = spearmanr(tr, te).correlation
    print(f"\n全库 train_ic vs test_ic 秩相关 = {rho:+.3f}")
    print("（>0：样本内 IC 能预测样本外，train 选有效；≈0：纯噪声，整个挖法白搭）")


if __name__ == "__main__":
    main()
