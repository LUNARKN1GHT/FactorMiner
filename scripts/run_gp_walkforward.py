"""Walk-forward 样本外验证：滚动训练-测试，逐折进化 + 逐个体检。

每折在自己的训练段重跑 NSGA-II 进化出一条 Pareto 前沿，
再在该折没碰过的测试段逐个体检 train|ICIR| vs test|ICIR|。
最后跨折汇总「每折按训练段最强挑出的因子」的样本外战绩，看它稳不稳。
"""

import sys
from collections.abc import Callable

import pandas as pd
import yaml

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.engine import GPConfig
from src.gp.evaluator import evaluate, to_wide
from src.gp.nsga2 import run_gp_nsga2
from src.gp.tree import Node
from src.utils.logger import setup_experiment_logger


def make_rolling_windows(
    splits: list[pd.Timestamp],
    train_span: pd.DateOffset,
    test_span: pd.DateOffset,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """滚动窗口：每个切点向后回看 train_span 当训练段、向前取 test_span 当测试段。

    Args:
        splits: 每折训练/测试的分界点（train 上界 = test 下界）。
        train_span: 训练段长度（定长，起点随切点前滑）。
        test_span: 测试段长度。

    Returns:
        每折一组 (train_lo, split, test_hi)；半开区间 train=[train_lo, split)、test=[split, test_hi)。
    """
    windows = []
    for split in splits:
        windows.append((split - train_span, split, split + test_span))
    return windows


def make_objective(
    wide: dict[str, pd.DataFrame], fwd_train: pd.DataFrame, method: str
) -> Callable[[Node], float | None]:
    """构造 NSGA 的单目标函数：本折训练段 |ICIR|，求值失败返回 None。

    在 main 的 fold 循环里每折调一次——把当折的 fwd_train 绑进闭包，
    既避免「循环里定义闭包捕获循环变量」的坑，也和 fitness.make_fitness 同一套路。
    """

    def objective_fn(tree):
        try:
            ic = calc_ic_series(evaluate(tree, wide), fwd_train, method=method).dropna()
        except Exception:
            return None
        if len(ic) < 10:
            return None
        icir = calc_icir(ic)
        return abs(icir) if pd.notna(icir) else None

    return objective_fn


def main(config_path: str = "configs/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    gp_cfg = GPConfig(**cfg["gp"])
    method = cfg["evaluation"]["ic_method"]

    logger, _ = setup_experiment_logger()

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    splits = [pd.Timestamp(d) for d in ("2021-01-01", "2022-01-01", "2023-01-01")]
    windows = make_rolling_windows(
        splits,
        train_span=pd.DateOffset(years=3),
        test_span=pd.DateOffset(years=1),
    )

    picks: list = []  # 每折「按 train|ICIR| 最强挑出的因子」的战绩，留作跨折汇总

    for fold, (train_lo, split, test_hi) in enumerate(windows, start=1):
        # 滚动窗的训练段要同时卡上下界（半开）；holdout 只卡上界，这里多了 >= train_lo
        fwd_train = fwd.loc[(fwd.index >= train_lo) & (fwd.index < split)]

        logger.info("")
        logger.info(
            "=== fold%d | train [%s, %s) %d 天 | test [%s, %s) ===",
            fold,
            train_lo.date(),
            split.date(),
            len(fwd_train),
            split.date(),
            test_hi.date(),
        )

        # 搜索目标只喂本折训练段 → GP 全程不接触该折测试段
        pareto = run_gp_nsga2(make_objective(wide, fwd_train, method), gp_cfg)

        # 逐个体检：同一因子在本折 train / test 两段的 |ICIR|
        logger.info("%4s %11s %10s %6s  expr", "size", "train|ICIR|", "test|ICIR|", "衰减")
        best: tuple[int, float, float, float, Node] | None = None
        for tree, _, size in pareto:
            ic = calc_ic_series(evaluate(tree, wide), fwd, method=method)
            tr = abs(calc_icir(ic.loc[(ic.index >= train_lo) & (ic.index < split)].dropna()))
            te = abs(calc_icir(ic.loc[(ic.index >= split) & (ic.index < test_hi)].dropna()))
            decay = (1 - te / tr) * 100 if tr else 0.0
            logger.info("%4d %11.4f %10.4f %5.0f%%  %s", size, tr, te, decay, tree)
            if best is None or tr > best[1]:
                best = (size, tr, te, decay, tree)

        if best is not None:
            picks.append((fold, *best))

    # 跨折汇总：模拟「每折都挑当折样本内最强因子」后，样本外到底稳不稳
    logger.info("")
    logger.info("=== 跨折汇总：每折按 train|ICIR| 选出的因子，其样本外战绩 ===")
    logger.info("%4s %4s %11s %10s %6s  expr", "fold", "size", "train|ICIR|", "test|ICIR|", "衰减")
    for fold, size, tr, te, decay, tree in picks:
        logger.info("%4d %4d %11.4f %10.4f %5.0f%%  %s", fold, size, tr, te, decay, tree)

    te_list = [te for *_, te, _, _ in picks]
    logger.info(
        "OOS |ICIR| 各折=%s | 均值=%.4f",
        [round(x, 4) for x in te_list],
        sum(te_list) / len(te_list),
    )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
