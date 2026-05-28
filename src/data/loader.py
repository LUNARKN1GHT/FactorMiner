"""A 股行情数据加载模块。"""

import pandas as pd


def load_daily_prices(
    codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """加载日频行情数据。

    返回 MultiIndex DataFrame，index=(date, code)，
    columns=[open, high, low, close, volume, amount]。
    """
    # TODO: 对接 akshare 或其他数据源
    raise NotImplementedError


def load_universe(name: str, date: str) -> list[str]:
    """获取指定日期的股票池成分股列表。"""
    # TODO: 对接指数成分股数据
    raise NotImplementedError
