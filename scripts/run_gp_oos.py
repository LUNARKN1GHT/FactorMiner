"""样本外验证：只在训练段进化，在测试段检验因子。"""

import sys

import pandas as pd
import yaml

from src.data.preprocess import to_panel
from src.evaluation.fitness import make_fitness
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig, run_gp
from src.gp.evaluator import evaluate, to_wide
from src.utils.logger import setup_experiment_logger


def main(config_path: str = "configs/default.yaml", split: str = "2022-01-01") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    gp_cfg = GPConfig(**cfg["gp"])
    method = cfg["evaluation"]["ic_method"]

    logger, log_dir = setup_experiment_logger()

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")

    split_ts = pd.Timestamp(split)
    fwd_train = fwd.loc[fwd.index < split_ts]
    logger.info(
        "切点 %s | 训练段 %d 天 | 测试段 %d 天",
        split_ts.date(),
        len(fwd_train),
        len(fwd) - len(fwd_train),
    )

    # 1. 适应度只喂训练段
    fitness_fn = make_fitness(wide=wide, forward_returns=fwd_train, method=method)
    results = run_gp(fitness_fn=fitness_fn, config=gp_cfg, logger=logger, log_dir=log_dir)
    best_tree, _ = results[0]

    # 2. 同一个因子，分别在训练/测试段计算 ICIR
    ic = calc_ic_series(evaluate(node=best_tree, wide=wide), return_df=fwd, method=method)
    icir_tr = abs(calc_icir(ic.loc[ic.index < split_ts].dropna()))
    icir_te = abs(calc_icir(ic.loc[ic.index >= split_ts].dropna()))

    logger.info("=== 样本外验证 ===")
    logger.info("最佳因子: %s", best_tree)
    logger.info("训练段 |ICIR| = %.4f", icir_tr)
    logger.info("测试段 |ICIR| = %.4f", icir_te)
    logger.info("衰减 = %.0f%%", (1 - icir_te / icir_tr) * 100 if icir_tr else 0.0)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
