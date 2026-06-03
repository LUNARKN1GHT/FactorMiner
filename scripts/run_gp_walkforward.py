"""Walk-forward 样本外验证：滚动训练-测试，逐折进化 + 逐个体检。

第 1 步：先只把滚动窗口算出来、打印验证，不接 GP。
确认三折区间无误后，再往 main 里塞「进化 + 逐个体检」。
"""

import sys

import pandas as pd

from src.utils.logger import setup_experiment_logger


def make_rolling_windows(
    splits: list[pd.Timestamp],
    train_span: pd.DateOffset,
    test_span: pd.DateOffset,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """滚动窗口：每个切点向后回看 train_span 当训练段、向前取 test_span 当测试段。

    与扩张窗的唯一区别——训练段定长，起点 train_lo 跟着切点一起前滑。

    Args:
        splits: 每折训练/测试的分界点（train 的上界 = test 的下界）。
        train_span: 训练段长度（定长）。
        test_span: 测试段长度。

    Returns:
        每折一组 (train_lo, split, test_hi) 三个时间戳；
        区间约定半开：train=[train_lo, split)，test=[split, test_hi)。
    """
    windows = []
    for split in splits:
        train_lo = split - train_span  # 训练段下界：切点往回数 train_span
        test_hi = split + test_span  # 测试段上界：切点往前数 test_span
        windows.append((train_lo, split, test_hi))
    return windows


def main(config_path: str = "configs/default.yaml") -> None:
    logger, _ = setup_experiment_logger()

    # 先手写死三个切点（= 三折）；之后可改成按数据年份自动生成
    splits = [pd.Timestamp(d) for d in ("2021-01-01", "2022-01-01", "2023-01-01")]
    windows = make_rolling_windows(
        splits,
        train_span=pd.DateOffset(years=3),
        test_span=pd.DateOffset(years=1),
    )

    logger.info("=== 滚动窗口（共 %d 折，3 年训练 / 1 年测试）===", len(windows))
    for i, (train_lo, split, test_hi) in enumerate(windows, start=1):
        logger.info(
            "fold%d | train [%s, %s) | test [%s, %s)",
            i,
            train_lo.date(),
            split.date(),
            split.date(),
            test_hi.date(),
        )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(config)
