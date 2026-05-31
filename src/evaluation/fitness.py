"""适应度分析：今天的因子值，是否能预测未来的收益"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import calc_ic_series, calc_icir
from src.gp.evaluator import evaluate


def compute_forward_returns(close: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """计算前向收益率

    Args:
        close (pd.DataFrame): date x code 宽表（复权收盘价）
        period (int, optional): _description_. 前视收益率的窗口长度.

    Returns:
        pd.DataFrame: date x code，收益率表
    """
    # pct_change(period) 在第 t 行算的是「过去 period 日」收益: close_t / close_{t-period} - 1
    # 再 shift(-period) 把数值整体往「上」挪 period 行：
    #   第 t 行就装上了原本第 t+period 行的值 = close_{t+period}/close_t - 1，正是未来收益
    return close.pct_change(period).shift(-period)


def make_fitness(wide: pd.DataFrame, forward_returns, method="rank") -> float:
    """构造适应度函数：表达式树 → 适应度分数（这里用 |ICIR|）。

    Args:
        wide (_type_): _description_
        forward_returns (_type_): _description_
        method (str, optional): _description_. Defaults to "rank".

    Returns:
        float: _description_
    """

    def fitness(node) -> float:
        try:
            factor = evaluate(node=node, wide=wide)
        except Exception:
            return -1.0

        ic = calc_ic_series(factor, forward_returns, method=method).dropna()
        if len(ic) < 10:
            return -1.0

        icir = calc_icir(ic)
        if not np.isfinite(icir):
            return -1.0

        return abs(icir)

    return fitness  # type: ignore
