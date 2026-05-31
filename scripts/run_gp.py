"""GP 因子挖掘主入口脚本。"""

import sys

import yaml

from src.data.loader import load_daily_prices, load_universe_cached
from src.evaluation.fitness import compute_forward_returns, make_fitness
from src.gp.engine import GPConfig, run_gp
from src.gp.evaluator import to_wide


def main(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gp_cfg = GPConfig(**cfg["gp"])
    dcfg, ecfg, bcfg = cfg["data"], cfg["evaluation"], cfg["backtest"]

    # 1. 加载行情 —> 转宽表
    codes = load_universe_cached(dcfg["universe"])
    prices = load_daily_prices(
        codes=codes, start_date=dcfg["start_date"], end_date=dcfg["end_date"]
    )
    wide = to_wide(prices=prices)

    # 2. 未来收益
    fwd = compute_forward_returns(wide["close"], period=bcfg["holding_period"])

    # 3. 适应度函数
    fitness_fn = make_fitness(wide=wide, forward_returns=fwd, method=ecfg["ic_method"])

    # 4. 开始进化
    results = run_gp(fitness_fn=fitness_fn, config=gp_cfg)

    # 5. 打印 Top 因子
    print("\n === Top 10 因子 ===")
    for tree, score in results[:10]:
        print(f"|ICIR| = {score:.4f} {tree}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
