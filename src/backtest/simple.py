"""基于因子值的简单分组回测（不重叠调仓 + 换手手续费 + 风险指标）。"""

import numpy as np
import pandas as pd


def _tradable_masks(close: pd.DataFrame, limit: float = 0.098) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从收盘价面板计算可买可卖掩码

    Args:
        close (pd.DataFrame): 收盘价
        limit (float, optional): 单日最大涨幅. Defaults to 0.098.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 单日是否存在涨停
    """
    pct = close.pct_change()
    suspended = close.isna()
    can_by = ~((pct >= limit) | suspended)
    can_sell = ~((pct <= -limit) | suspended)
    return can_by, can_sell


def _metrics(net: pd.Series, turnover: pd.Series, ppy: float) -> dict:
    """从一条单期净收益序列算净值与风险指标。"""
    equity = (1 + net).cumprod()
    return {
        "equity": equity,
        "ann_return": _annualized(equity, ppy),
        "ann_vol": net.std() * np.sqrt(ppy),
        "sharpe": _sharpe(net, ppy),
        "max_drawdown": _max_drawdown(equity),
        "avg_turnover": turnover.mean(),
        "win_rate": (net > 0).mean(),
    }


def run_backtest(
    factor_df: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    close: pd.DataFrame | None = None,
    holding_period: int = 5,
    n_groups: int = 5,
    commission: float = 0.001,
) -> dict:
    """分组多空回测：每 holding_period 天调仓一次（不重叠），按换手扣手续费。

    启用涨跌停过滤

    Args:
        factor_df: index=date, columns=code 的因子值矩阵。
        fwd_ret: 同结构的未来收益矩阵。
        close: 收盘价表
        holding_period: 持仓周期长度
        n_groups: 分组数。
        commission: 单边手续费率。

    Returns:
        包含各组累计收益、多空组合收益等的字典。
    """
    dates = factor_df.index.intersection(fwd_ret.index)
    rebal_dates = dates[::holding_period]  # 每隔 N 天一个调仓日

    can_by = _tradable_masks(close)[0] if close is not None else None

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    records = []

    for d in rebal_dates:
        f = factor_df.loc[d].dropna()
        if len(f) < n_groups * 2:
            # 可选票太少，跳过这次调仓
            continue

        labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
        long_codes = set(f.index[labels == labels.max()])  # 因子最高组
        short_codes = set(f.index[labels == 0])  # 因子最低组

        # 约束：涨停停牌不能进行操作
        if can_by is not None and d in can_by.index:
            buyable = set(can_by.loc[d][can_by.loc[d]].index)
            long_codes &= buyable

        r: pd.Series = fwd_ret.loc[d]
        long_ret = r.reindex(long_codes).mean()
        short_ret = r.reindex(short_codes).mean()
        bench = r.reindex(f.index).mean()  # 全池基准：当日可选股票等权

        gross = long_ret - short_ret  # 理论多空
        long_excess = long_ret - bench  # 现实多头超额
        if np.isnan(gross):  # 末尾几期 fwd_ret 全 NaN → 不是真实持有期，跳过
            continue

        long_turn = _turnover(prev_long, long_codes)
        short_turn = _turnover(prev_short, short_codes)

        records.append(
            {
                "date": d,
                "ls_net": gross - (long_turn + short_turn) * commission * 2,
                "lo_net": long_excess - long_turn * commission * 2,
                "turnover": long_turn + short_turn,
                "long_turn": long_turn,
                "n_long": len(long_codes),
            }
        )
        prev_long, prev_short = long_codes, short_codes

    if not records:
        raise ValueError("没有有效调仓日，检查 factor / fwd_ret 是否对齐")
    bt = pd.DataFrame(records).set_index("date")
    ppy = 252 / holding_period
    return {
        "table": bt,
        "long_short": _metrics(bt["ls_net"], bt["turnover"], ppy),  # 理论：含不可实现空头
        "long_only": _metrics(bt["lo_net"], bt["long_turn"], ppy),  # 现实：可买多头 vs 全池
    }


def _turnover(prev: set[str], cur: set[str]) -> float:
    """新一期持仓里，有多少比例是新换进来的（单腿、单向）。"""
    if not cur:
        return 0.0
    return len(cur - prev) / len(cur)


def _annualized(equity: pd.Series, ppy: float) -> float:
    n_years = len(equity) / ppy
    return equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0


def _sharpe(returns: pd.Series, ppy: float, rf: float = 0.0) -> float:
    excess = returns - rf / ppy
    return excess.mean() / excess.std() * np.sqrt(ppy) if excess.std() > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    """最大回撤：净值相对历史高点的最大跌幅（负数）。"""
    peak = equity.cummax()
    return (equity / peak - 1).min()
