"""适应度分析：今天的因子值，是否能预测未来的收益"""

import pandas as pd


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
