"""STGP 接入 NSGA 的小种群 smoke：跑通 + 抽查 Pareto 因子全部类型合法。"""

import sys

import pandas as pd
import yaml

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig
from src.gp.evaluator import evaluate, to_wide
from src.gp.nsga2 import run_gp_nsga2
from src.gp.stgp import recompute_type


def main(config_path: str = "configs/smoke.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    def objective_fn(tree):
        try:
            ic = calc_ic_series(evaluate(tree, wide), fwd, method=method).dropna()
        except Exception:
            return None
        if len(ic) < 10:
            return None
        icir = calc_icir(ic)
        return abs(icir) if pd.notna(icir) else None

    # gp 块由 smoke.yaml 驱动（含 strongly_typed: true）
    gp_cfg = GPConfig(**cfg["gp"])
    pareto = run_gp_nsga2(objective_fn, gp_cfg)

    print(f"=== STGP Pareto 前沿（{len(pareto)} 个）===")
    bad = 0
    for tree, icir, size in pareto:
        try:
            recompute_type(tree)  # 抽查：进化产物是否仍类型合法
        except AssertionError as exc:
            bad += 1
            print("非法：", exc)
        print(f"[{tree.out_type}] |ICIR|={icir:.4f} size={size:2d}  {tree}")
    print(f"非法 {bad} 例")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml"
    main(config)
