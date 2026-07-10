"""把本项目 tushare/hs300 数据伪装成 AlphaGen 的 StockData 接口。

baostock 的私有协议端口（10030）被计算节点防火墙挡死、qlib 官方数据源也被禁，服务器上的
qlib/baostock 数据链路走不通。退而求其次：`Expression.evaluate(data, period)`
（见 `alphagen/data/expression.py`）只依赖 `data.data`（`(days, features, stocks)` 张量）、
`data.max_backtrack_days`、`data.max_future_days`、`data.n_days`、`data.n_stocks` 这几个
鸭子类型属性，不牵扯任何 qlib 内部实现——直接用本项目已有的 `data/cache/prices_clean.parquet`
构造同接口对象，就能复用 AlphaGen 全部算子求值逻辑，完全绕开 qlib/baostock。

代价：不再是官方 CSI300(qlib/baostock) 口径，而是跟本项目 GP/RL/LLM/QuantFactor
同一份 tushare/hs300 数据——不再需要事后"桥接"翻译，天然可比。
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PARQUET = _PROJECT_ROOT / "data" / "cache" / "prices_clean.parquet"

# 跟 alphagen_qlib.stock_data.FeatureType 的枚举顺序一一对应：OPEN,CLOSE,HIGH,LOW,VOLUME,VWAP
_FEATURE_COLUMNS = ["open", "close", "high", "low", "volume", "vwap"]


class TushareStockData:
    def __init__(
        self,
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        parquet_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        path = parquet_path or str(_DEFAULT_PARQUET)
        prices = pd.read_parquet(path)
        if "vwap" not in prices.columns:
            prices = prices.copy()
            prices["vwap"] = prices["amount"] / prices["volume"].replace(0, np.nan)

        dates_all = prices.index.get_level_values(0).unique().sort_values()
        start_idx = dates_all.searchsorted(pd.Timestamp(start_time))
        end_idx = dates_all.searchsorted(pd.Timestamp(end_time))
        lo = max(0, start_idx - max_backtrack_days)
        hi = min(len(dates_all) - 1, end_idx + max_future_days)
        window_dates = dates_all[lo:hi + 1]

        level0 = prices.index.get_level_values(0)
        mask = (level0 >= window_dates[0]) & (level0 <= window_dates[-1])
        sub = prices.loc[mask, _FEATURE_COLUMNS]

        panels = []
        stock_ids = None
        for col in _FEATURE_COLUMNS:
            wide = sub[col].unstack(level=1).reindex(index=window_dates)
            if stock_ids is None:
                stock_ids = wide.columns
            else:
                wide = wide.reindex(columns=stock_ids)
            panels.append(wide.values)

        arr = np.stack(panels, axis=1)  # (days, 6, stocks)
        self.data = torch.tensor(arr, dtype=torch.float, device=device)
        self._dates = window_dates
        self._stock_ids = stock_ids
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.device = device

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids
