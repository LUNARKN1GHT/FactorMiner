"""A 股行情数据加载模块，带逐股 Parquet 缓存和断点续传。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"

_PRICE_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

_INDEX_CODE_MAP = {
    "hs300": "000300",
    "zz500": "000905",
    "sz50": "000016",
}


def _stock_cache_path(code: str, start_date: str, end_date: str, stock_cache_dir: Path) -> Path:
    return stock_cache_dir / f"{code}_{start_date}_{end_date}.parquet"


def _fetch_single_stock(code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """拉取单支股票前复权日频数据，失败指数退避重试。"""
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.debug("拉取 %s 第 %d 次失败，%.0fs 后重试：%s", code, attempt + 1, wait, e)
            time.sleep(wait)

    df = df.rename(columns=_PRICE_COL_MAP)
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if c in df.columns]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    return df


def load_daily_prices(
    codes: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path | str | None = None,
    request_interval: float = 0.5,
) -> pd.DataFrame:
    """加载日频行情数据（前复权），逐股缓存 + 断点续传。

    返回 MultiIndex DataFrame，index=(date, code)，
    columns=[open, high, low, close, volume, amount, turnover]。

    每只股票单独缓存到 data/cache/stocks/ 下，中途中断下次自动跳过已完成的。

    Parameters
    ----------
    codes:
        股票代码列表，6 位数字代码，如 ["000001", "600000"]。
    start_date / end_date:
        日期字符串，格式 "YYYY-MM-DD"。
    cache_dir:
        缓存根目录，默认 data/cache/。
    request_interval:
        每次请求间隔（秒），默认 0.5s。
    """
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    stock_cache_dir = cache_dir / "stocks"
    stock_cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing = []

    for code in codes:
        p = _stock_cache_path(code, start_date, end_date, stock_cache_dir)
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append(code)

    if missing:
        logger.info("已缓存 %d 只，待拉取 %d 只", len(codes) - len(missing), len(missing))
        for i, code in enumerate(missing):
            try:
                df = _fetch_single_stock(code, start_date, end_date)
                p = _stock_cache_path(code, start_date, end_date, stock_cache_dir)
                df.to_parquet(p)
                frames.append(df)
            except Exception as e:
                logger.warning("拉取 %s 失败（跳过）：%s", code, e)
            if i < len(missing) - 1:
                time.sleep(request_interval)
            if (i + 1) % 50 == 0:
                logger.info("进度 %d / %d", len(codes) - len(missing) + i + 1, len(codes))
    else:
        logger.info("全部 %d 只命中缓存", len(codes))

    if not frames:
        raise RuntimeError("没有任何数据，请检查网络或 akshare 版本。")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index(["date", "code"]).sort_index()
    return result


def load_universe(name: str, date: str | None = None) -> list[str]:
    """获取指数成分股列表（实时）。

    Parameters
    ----------
    name:
        股票池名称，支持 "hs300" / "zz500" / "sz50"。
    date:
        暂未使用，预留给历史成分股查询。
    """
    index_code = _INDEX_CODE_MAP.get(name.lower())
    if index_code is None:
        raise ValueError(f"不支持的股票池：{name}，可选：{list(_INDEX_CODE_MAP)}")

    df = ak.index_stock_cons(symbol=index_code)
    codes = df["品种代码"].tolist() if "品种代码" in df.columns else df.iloc[:, 0].tolist()
    return [str(c).zfill(6) for c in codes]


def load_universe_cached(
    name: str,
    cache_dir: Path | str | None = None,
    ttl_days: int = 7,
) -> list[str]:
    """获取成分股列表，带 TTL 文件缓存（默认 7 天刷新）。"""
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"universe_{name}.txt"

    if cache_file.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit="s")).days
        if age_days < ttl_days:
            logger.info("成分股缓存命中（%d 天前）", age_days)
            return cache_file.read_text().splitlines()

    codes = load_universe(name)
    cache_file.write_text("\n".join(codes))
    logger.info("成分股已缓存：%d 只", len(codes))
    return codes
