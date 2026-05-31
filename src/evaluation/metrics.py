"""因子评估指标。"""

import numpy as np
import pandas as pd
from scipy import stats


def calc_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = "rank",
) -> float:
    """计算信息系数 (IC)。

    Args:
        factor_values: 因子值。
        forward_returns: 未来收益。
        method: "rank" 为 Rank IC (Spearman), "normal" 为 Pearson IC。
    """
    mask = factor_values.notna() & forward_returns.notna()
    fv = factor_values[mask]
    fr = forward_returns[mask]

    if len(fv) < 10:
        return np.nan

    # 截面退化：因子或收益在该日是常数 -> 相关系数无定义，直接判 NaN
    if fv.nunique() <= 1 or fr.nunique() <= 1:
        return np.nan

    if method == "rank":
        corr, _ = stats.spearmanr(fv, fr)
    else:
        corr, _ = stats.pearsonr(fv, fr)

    return corr


def calc_ic_series(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    method: str = "rank",
) -> pd.Series:
    """按日期计算 IC 序列。

    Args:
        factor_df: index=date, columns=code 的因子值矩阵。
        return_df: 同结构的未来收益矩阵。
    """
    dates = factor_df.index.intersection(return_df.index)
    ic_list = {}
    for date in dates:
        ic_list[date] = calc_ic(factor_df.loc[date], return_df.loc[date], method)
    return pd.Series(ic_list)


def calc_icir(ic_series: pd.Series) -> float:
    """IC 信息比率 = mean(IC) / std(IC)。"""
    if ic_series.std() == 0:
        return 0.0
    return ic_series.mean() / ic_series.std()


def group_returns(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """分组回测：按因子值分组，计算各组平均收益。

    Returns:
        DataFrame, index=date, columns=[G1, G2, ..., Gn]（G1 为因子值最小组）。
    """
    dates = factor_df.index.intersection(return_df.index)
    result = {}
    for date in dates:
        fv = factor_df.loc[date].dropna()
        fr = return_df.loc[date]
        labels = pd.qcut(fv, n_groups, labels=False, duplicates="drop")
        group_ret = {}
        for g in range(n_groups):
            codes = labels[labels == g].index
            group_ret[f"G{g + 1}"] = fr.reindex(codes).mean()
        result[date] = group_ret
    return pd.DataFrame(result).T
