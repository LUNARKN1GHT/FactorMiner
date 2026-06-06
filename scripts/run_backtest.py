"""现实约束回测：拿 walkforward.pkl 里挑出的因子，对比「理论多空」vs「现实多头」。"""

import pickle
import sys

import pandas as pd
import yaml

from src.backtest.simple import run_backtest
from src.data.preprocess import to_panel
from src.gp.evaluator import evaluate, to_wide
from src.utils.logger import setup_experiment_logger


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)

    with open(payload["config_path"]) as f:
        cfg = yaml.safe_load(f)
    hold = cfg["backtest"]["holding_period"]
    n_groups = cfg["evaluation"].get("n_groups", 5)
    comm = cfg["backtest"]["commission"]

    logger, _ = setup_experiment_logger()

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    close = to_panel(prices, "close")  # 涨跌停/停牌过滤用

    for fold, _size, _tr, _te, _decay, tree, *_ in payload["picks"]:
        factor = evaluate(tree, wide)
        res = run_backtest(
            factor, fwd, close=close, holding_period=hold, n_groups=n_groups, commission=comm
        )
        logger.info("")
        logger.info("=== fold%d  %s ===", fold, tree)
        for label, key in (("理论多空", "long_short"), ("现实多头", "long_only")):
            m = res[key]
            logger.info(
                "  %-8s sharpe=%6.3f 年化=%7.2f%% 最大回撤=%7.2f%% 平均换手=%.2f",
                label,
                m["sharpe"],
                m["ann_return"] * 100,
                m["max_drawdown"] * 100,
                m["avg_turnover"],
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/run_backtest.py <walkforward.pkl 路径>")
    main(sys.argv[1])
