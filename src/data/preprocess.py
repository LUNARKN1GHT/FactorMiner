"""行情数据清洗与预处理。

从 load_daily_prices() 返回的 MultiIndex DataFrame 开始，输出可直接用于
因子评估和 GP 适应度计算的面板数据。

典型用法：
    from data.loader import load_daily_prices
    from data.preprocess import clean_prices, compute_returns, to_panel

    raw = load_daily_prices(codes, "2018-01-01", "2024-12-31")
    prices = clean_prices(raw)
    returns = compute_returns(prices, period=5)
    close_panel = to_panel(prices, "close")   # date × code
    ret_panel   = to_panel(returns, "fwd_ret")
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 停牌判断：成交量为 0 视为停牌
_SUSPENDED_VOL_THRESH = 0.0


def clean_prices(
    df: pd.DataFrame,
    min_trading_days: int = 252,
    fill_limit: int = 3,
) -> pd.DataFrame:
    """清洗多股票行情数据。

    处理步骤：
    1. 去重：同一 (date, code) 保留首条
    2. 停牌日：volume == 0 → 所有价格列置 NaN
    3. 价格前向填充（限 fill_limit 天）
    4. 剔除历史不足 min_trading_days 的股票
    5. 删除全为 NaN 的日期行

    Parameters
    ----------
    df:
        load_daily_prices() 返回的 MultiIndex(date, code) DataFrame。
    min_trading_days:
        保留股票所需的最少非停牌交易日，默认 252（约一年）。
    fill_limit:
        价格前向填充的最大连续天数，超出则保留 NaN。

    Returns
    -------
    清洗后的 MultiIndex(date, code) DataFrame，列同输入。
    """
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    # 1. 去重
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dupes = before - len(df)
    if dupes:
        logger.info("去重：删除 %d 条重复行", dupes)

    # 2. 停牌日 → 价格置 NaN
    if "volume" in df.columns:
        suspended = df["volume"] <= _SUSPENDED_VOL_THRESH
        n_susp = suspended.sum()
        if n_susp:
            df.loc[suspended, price_cols] = np.nan
            logger.info("停牌日处理：%d 条价格置 NaN", n_susp)

    # 3. 前向填充价格（按股票分组）
    df = df.groupby(level="code", group_keys=False).apply(lambda g: g.ffill(limit=fill_limit))

    # 4. 剔除历史不足的股票
    trading_days = df.groupby(level="code")["close"].count()
    valid_codes = trading_days[trading_days >= min_trading_days].index
    n_drop = df.index.get_level_values("code").nunique() - len(valid_codes)
    if n_drop:
        logger.info("剔除历史不足 %d 日的股票 %d 只", min_trading_days, n_drop)
    df = df[df.index.get_level_values("code").isin(valid_codes)]

    # 5. 删除全为 NaN 的日期（通常不会有，保险起见）
    df = df.dropna(how="all")

    logger.info(
        "清洗完成：%d 只股票，%d 条记录，日期 %s ~ %s",
        df.index.get_level_values("code").nunique(),
        len(df),
        df.index.get_level_values("date").min().date(),
        df.index.get_level_values("date").max().date(),
    )
    return df


def compute_returns(
    df: pd.DataFrame,
    period: int = 5,
    col: str = "close",
) -> pd.DataFrame:
    """计算 N 日 forward return，存入新列 fwd_ret。

    fwd_ret[t] = close[t+period] / close[t] - 1

    Parameters
    ----------
    df:
        clean_prices() 返回的 MultiIndex DataFrame，含 close 列。
    period:
        前向收益窗口（交易日）。
    col:
        用于计算收益的价格列，默认 close。

    Returns
    -------
    附加 fwd_ret 列的 DataFrame（原地修改副本）。
    """
    df = df.copy()
    # 每只股票独立计算 shift，避免跨股票污染
    fwd = df[col].groupby(level="code").transform(lambda s: s.shift(-period) / s - 1)
    df["fwd_ret"] = fwd
    logger.info("已计算 %d 日 forward return，列名 fwd_ret", period)
    return df


def to_panel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """将 MultiIndex(date, code) 的单列 unstack 为 date × code 矩阵。

    Parameters
    ----------
    df:
        MultiIndex DataFrame。
    col:
        需要 pivot 的列名。

    Returns
    -------
    index=date，columns=code 的 DataFrame。
    """
    return df[col].unstack(level="code")
