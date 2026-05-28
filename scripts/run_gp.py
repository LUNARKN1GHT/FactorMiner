"""GP 因子挖掘主入口脚本。"""

import sys

import yaml

sys.path.insert(0, ".")

from src.gp.engine import GPConfig


def main(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gp_cfg = GPConfig(**cfg["gp"])

    # TODO: 加载真实数据，构建适应度函数
    # data = load_daily_prices(...)
    # def fitness_fn(tree): ...

    print("GP 因子挖掘引擎已就绪。")
    print(
        f"配置: population={gp_cfg.population_size}, "
        f"generations={gp_cfg.generations}, "
        f"max_depth={gp_cfg.max_depth}"
    )
    print("等待数据接口对接后启动进化...")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
