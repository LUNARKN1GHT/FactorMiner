"""GP 因子挖掘主入口脚本。"""

import pickle
import sys

import pandas as pd
import yaml

from src.core.evaluator import to_wide
from src.data.preprocess import to_panel
from src.evaluation.fitness import make_fitness
from src.gp.engine import GPConfig, run_gp
from src.utils.logger import setup_experiment_logger


def main(config_path: str = "configs/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gp_cfg = GPConfig(**cfg["gp"])
    dcfg, ecfg = cfg["data"], cfg["evaluation"]

    logger, log_dir = setup_experiment_logger(
        tag="Simple_gp", meta={"config": config_path, "gp": cfg["gp"]}
    )
    logger.info("配置文件: %s", config_path)

    # 1. 读已清晰的面板数据
    clean_path = dcfg.get("clean_path", "data/cache/prices_clean.parquet")
    prices = pd.read_parquet(clean_path)
    logger.info("载入 %s: shape=%s, 列=%s", clean_path, prices.shape, list(prices.columns))
    if "fwd_ret" not in prices.columns:
        raise KeyError("prices_clean 缺少 fwd_ret 列，先运行 preprocess_data.py")

    # 2. 价格字段 -> 宽表字典
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))  # 去掉 fwd_ret，避免被当成特征
    fwd = to_panel(prices, "fwd_ret")  # 未来收益

    # 3. 适应度函数
    fitness_fn = make_fitness(
        wide=wide,
        forward_returns=fwd,
        method=ecfg["ic_method"],
        parsimony=ecfg.get("parsimony", 0.0),
    )
    results = run_gp(fitness_fn=fitness_fn, config=gp_cfg, logger=logger, log_dir=log_dir)

    # 4. 打印 Top 因子
    logger.info("=== Top 10 因子 ===")
    for tree, score in results[:10]:
        logger.info("|ICIR|=%.4f  %s", score, tree)

    # 持久化存储因子
    with open(log_dir / "top_factors.pkl", "wb") as fh:
        pickle.dump(results, fh)
    logger.info("Top 因子已存: %s", log_dir / "top_factors.pkl")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
