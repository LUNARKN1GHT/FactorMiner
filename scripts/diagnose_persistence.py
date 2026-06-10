"""
因子持续性诊断：库里每个因子逐年 IC，看年×年排名相关是否稳定。
普遍<0 → 效力 regime 翻转坐实；≈0 → 噪声；>0 → 可选。
"""

import pickle

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.gp.evaluator import evaluate, to_wide
from src.gp.tree import Node


def year_ics(
    tree: Node, wide: dict[str, pd.DataFrame], fwd: pd.DataFrame, method: str, years: list[int]
) -> dict[int, float] | None:
    try:
        ic = calc_ic_series(evaluate(tree, wide), fwd, method=method)
    except Exception:
        return None
    out: dict[int, float] = {}
    for y in years:
        s = ic[ic.index.year == y].dropna()
        out[y] = float(s.mean()) if len(s) >= 20 else np.nan
    return out


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        lib = pickle.load(fh)
    with open(lib["config_path"]) as f:
        cfg = yaml.safe_load(f)

    factors = lib["factors"]
    method = cfg["evaluation"]["ic_method"]
    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    years = sorted(set(fwd.index.year))

    rows = Parallel(n_jobs=10, backend="loky", batch_size=32)(
        delayed(year_ics)(f["tree"], wide, fwd, method, years) for f in factors
    )
    m = pd.DataFrame([r for r in rows if r is not None])
    print("各年有效因子数:", m.notna().sum().to_dict())
    print("\n=== 年×年 因子 IC 秩相关（off-diagonal 看跨年持续性）===")
    print(m.corr(method="spearman").round(2).to_string())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/diagnose_persistence.py <factor_library.pkl 路径>")
    main(sys.argv[1])
