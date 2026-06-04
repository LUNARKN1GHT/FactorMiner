"""STGP 接入 NSGA 的小种群 smoke：跑通 + 抽查 Pareto 因子全部类型合法。"""

import pickle
import sys

import pandas as pd
import yaml

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig
from src.gp.evaluator import evaluate, to_wide
from src.gp.nsga2 import run_gp_nsga2
from src.gp.stgp import recompute_type
from src.utils.logger import setup_experiment_logger


def main(config_path: str = "configs/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]

    logger, log_dir = setup_experiment_logger()
    logger.info("STGP smoke | 配置：%s", config_path)

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

    # gp 块由 smoke.yaml 驱动（含 strongly_typed: true）；logger/log_dir 传进去，每代日志落 run.log
    gp_cfg = GPConfig(**cfg["gp"])
    pareto = run_gp_nsga2(objective_fn, gp_cfg, logger, log_dir)

    logger.info("=== STGP Pareto 前沿（%d 个）===", len(pareto))
    bad = 0
    for tree, icir, size in pareto:
        try:
            recompute_type(tree)  # 抽查：进化产物是否仍类型合法
        except AssertionError as exc:
            bad += 1
            logger.info("非法：%s", exc)
        logger.info("[%s] |ICIR|=%.4f size=%2d  %s", tree.out_type, icir, size, tree)
    logger.info("非法 %d 例", bad)

    with open(log_dir / "pareto_front.pkl", "wb") as fh:
        pickle.dump(pareto, fh)
    logger.info("Pareto 前沿已存：%s", log_dir / "pareto_front.pkl")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml"
    main(config)
