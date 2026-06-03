"""基于因子值的简单分组回测（不重叠调仓 + 换手手续费 + 风险指标）。"""

import numpy as np
import pandas as pd


def run_backtest(
    factor_df: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    holding_period: int = 5,
    n_groups: int = 5,
    commission: float = 0.001,
) -> dict:
    """分组多空回测：每 holding_period 天调仓一次（不重叠），按换手扣手续费。

    Args:
        factor_df: index=date, columns=code 的因子值矩阵。
        fwd_ret: 同结构的未来收益矩阵。
        fwd_red: 前向收益率表
        holding_period: 持仓周期长度
        n_groups: 分组数。
        commission: 单边手续费率。

    Returns:
        包含各组累计收益、多空组合收益等的字典。
    """
    dates = factor_df.index.intersection(fwd_ret.index)
    rebal_dates = dates[::holding_period]  # 每隔 N 天一个调仓日

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

        r: pd.Series = fwd_ret.loc[d]
        long_ret = r.reindex(long_codes).mean()
        short_ret = r.reindex(short_codes).mean()
        gross = long_ret - short_ret

        if np.isnan(gross):  # 末尾几期 fwd_ret 全 NaN → 不是真实持有期，跳过
            continue

        # 换手率
        turn = _turnover(prev_long, long_codes) + _turnover(prev_short, short_codes)
        cost = turn * commission * 2
        net = gross - cost

        records.append({"date": d, "gross": gross, "net": net, "turnover": turn})
        prev_long, prev_short = long_codes, short_codes

    if not records:
        raise ValueError("没有有效调仓日，检查 factor / fwd_ret 是否对齐")
    bt = pd.DataFrame(records).set_index("date")
    equity = (1 + bt["net"]).cumprod()  # 不重叠的复利才合法
    ppy = 252 / holding_period
    return {
        "table": bt,
        "equity": equity,
        "ann_return": _annualized(equity, ppy),
        "ann_vol": bt["net"].std() * np.sqrt(ppy),
        "sharpe": _sharpe(bt["net"], ppy),
        "max_drawdown": _max_drawdown(equity),
        "avg_turnover": bt["turnover"].mean(),
        "win_rate": (bt["net"] > 0).mean(),
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
