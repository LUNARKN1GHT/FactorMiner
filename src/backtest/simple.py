"""基于因子值的简单分组回测。"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import group_returns


def run_backtest(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    n_groups: int = 5,
    commission: float = 0.001,
) -> dict:
    """执行分组回测。

    Args:
        factor_df: index=date, columns=code 的因子值矩阵。
        return_df: 同结构的未来收益矩阵。
        n_groups: 分组数。
        commission: 单边手续费率。

    Returns:
        包含各组累计收益、多空组合收益等的字典。
    """
    gr = group_returns(factor_df, return_df, n_groups)

    gr_after_cost = gr - commission * 2  # 双边手续费

    cumulative = (1 + gr_after_cost).cumprod()
    long_short = gr_after_cost[f"G{n_groups}"] - gr_after_cost["G1"]
    ls_cumulative = (1 + long_short).cumprod()

    return {
        "group_returns": gr_after_cost,
        "cumulative": cumulative,
        "long_short": long_short,
        "ls_cumulative": ls_cumulative,
        "annual_return": _annualized_return(ls_cumulative),
        "sharpe": _sharpe_ratio(long_short),
    }


def _annualized_return(cumulative: pd.Series) -> float:
    total = cumulative.iloc[-1] / cumulative.iloc[0]
    n_years = len(cumulative) / 252
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def _sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / 252
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(252)
