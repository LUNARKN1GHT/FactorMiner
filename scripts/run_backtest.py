"""现实约束回测：从「纸面多空」逐层加现实约束，到「T+1 开盘成交的现实多头」。"""

import pickle
import sys

import pandas as pd
import yaml

from src.backtest.simple import run_backtest
from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
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
    open_p = to_panel(prices, "open")
    exec_ret = open_p.shift(-(1 + hold)) / open_p.shift(-1) - 1  # T + 1 开盘进、开盘出

    def _bt(ret: pd.DataFrame) -> dict:
        return run_backtest(
            factor_df=factor,
            fwd_ret=ret,
            close=close,
            holding_period=hold,
            n_groups=n_groups,
            commission=comm,
        )

    def _line(tag: str, m: dict) -> None:
        logger.info(
            "  %-18s sharpe=%6.3f 年化=%7.2f%% 最大回撤=%7.2f%%",
            tag,
            m["sharpe"],
            m["ann_return"] * 100,
            m["max_drawdown"] * 100,
        )

    for fold, _, _, _, _, tree, *_ in payload["picks"]:
        factor = evaluate(tree, wide)
        res_c, res_e = _bt(fwd), _bt(exec_ret)
        logger.info("")
        logger.info("=== fold%d. %s ===", fold, tree)
        _line("纸面多空·收盘T0", res_c["long_short"])  # 含不可实现空头 + 收盘即成交
        _line("现实多头·收盘T0", res_c["long_only"])  # 去空头 + 涨跌停过滤
        _line("现实多头·T+1开盘", res_e["long_only"])  # 再加 T+1 开盘成交


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/run_backtest.py <walkforward.pkl 路径>")
    main(sys.argv[1])
