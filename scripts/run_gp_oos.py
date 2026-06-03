"""样本外验证（多目标）：训练段进化出 Pareto 前沿，测试段逐个体检。"""

import sys

import pandas as pd
import yaml

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig
from src.gp.evaluator import evaluate, to_wide
from src.gp.nsga2 import run_gp_nsga2
from src.utils.logger import setup_experiment_logger


def main(config_path: str = "configs/default.yaml", split: str = "2022-01-01") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    gp_cfg = GPConfig(**cfg["gp"])
    method = cfg["evaluation"]["ic_method"]

    logger, log_dir = setup_experiment_logger()

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    split_ts = pd.Timestamp(split)
    fwd_train = fwd.loc[fwd.index < split_ts]  # 搜索只看训练段
    logger.info(
        "切点 %s | 训练段 %d 天 | 测试段 %d 天",
        split_ts.date(),
        len(fwd_train),
        len(fwd) - len(fwd_train),
    )

    # ① 搜索目标只用训练段收益 → GP 全程不接触测试段
    def objective_fn(tree):
        try:
            ic = calc_ic_series(evaluate(tree, wide), fwd_train, method=method).dropna()
        except Exception:
            return None
        if len(ic) < 10:
            return None
        icir = calc_icir(ic)
        return abs(icir) if pd.notna(icir) else None

    pareto = run_gp_nsga2(objective_fn, gp_cfg)

    # ② 整条前沿逐个体检：训练段 vs 测试段 ICIR
    logger.info("=== 样本外验证：Pareto 前沿逐个体检 ===")
    logger.info("%4s %11s %10s %6s  expr", "size", "train|ICIR|", "test|ICIR|", "衰减")
    for tree, _, size in pareto:
        ic = calc_ic_series(evaluate(tree, wide), fwd, method=method)
        tr = abs(calc_icir(ic.loc[ic.index < split_ts].dropna()))
        te = abs(calc_icir(ic.loc[ic.index >= split_ts].dropna()))
        decay = (1 - te / tr) * 100 if tr else 0.0
        logger.info("%4d %11.4f %10.4f %5.0f%%  %s", size, tr, te, decay, tree)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
