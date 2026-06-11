"""多目标 GP（NSGA-II）入口：在真实数据上挖一条 Pareto 前沿。"""

import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig
from src.gp.nsga2 import run_gp_nsga2
from src.utils.logger import setup_experiment_logger


def main(config_path: str = "configs/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gp_cfg = GPConfig(**cfg["gp"])
    dcfg = cfg["data"]

    logger, log_dir = setup_experiment_logger(
        tag="nsga2", meta={"config": config_path, "gp": cfg["gp"]}
    )
    logger.info("NSGA-II 多目标 GP | 配置：%s", config_path)

    clean_path = dcfg.get("clean_path", "data/cache/prices_clean.parquet")
    prices = pd.read_parquet(clean_path)
    logger.info("载入 %s：shape=%s", clean_path, prices.shape)

    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    # 单一目标值：|ICIR|（size 由 NSGA 内部当作第二目标，不在这里管）
    def objective_fn(tree):
        try:
            ic = calc_ic_series(evaluate(tree, wide), fwd).dropna()
        except Exception:
            return None
        if len(ic) < 10:
            return None
        icir = calc_icir(ic)
        return abs(icir) if np.isfinite(icir) else None

    pareto = run_gp_nsga2(objective_fn, gp_cfg)

    logger.info("=== 最终 Pareto 前沿（%d 个因子，按 size 排）===", len(pareto))
    for tree, icir, size in pareto:
        logger.info("|ICIR|=%.4f  size=%2d  %s", icir, size, tree)

    with open(log_dir / "pareto_front.pkl", "wb") as fh:
        pickle.dump(pareto, fh)
    logger.info("Pareto 前沿已存：%s", log_dir / "pareto_front.pkl")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
