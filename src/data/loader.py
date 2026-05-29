"""A 股行情数据加载模块，带 Parquet 本地缓存。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

# 默认缓存目录，相对于项目根
_CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"

# akshare 沪深300成分股接口返回的列名 → 标准列名
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

# HS300 指数代码（akshare 格式）
_INDEX_CODE_MAP = {
    "hs300": "000300",
    "zz500": "000905",
    "sz50": "000016",
}


def _cache_path(codes: list[str], start_date: str, end_date: str, cache_dir: Path) -> Path:
    """生成缓存文件路径，基于股票池 + 日期范围哈希。"""
    key = f"{'-'.join(sorted(codes)[:5])}_{len(codes)}stocks_{start_date}_{end_date}"
    # 文件名超长时截断，用 hash 保证唯一性
    if len(key) > 100:
        import hashlib

        h = hashlib.md5((",".join(sorted(codes))).encode()).hexdigest()[:8]
        key = f"{len(codes)}stocks_{h}_{start_date}_{end_date}"
    return cache_dir / f"{key}.parquet"


def _fetch_single_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """用 akshare 拉取单支股票前复权日频数据。"""

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",  # 前复权
    )
    df = df.rename(columns=_PRICE_COL_MAP)
    keep_cols = [
        c
        for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        if c in df.columns
    ]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    return df


def load_daily_prices(
    codes: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path | str | None = None,
    request_interval: float = 0.3,
) -> pd.DataFrame:
    """加载日频行情数据（前复权），带本地 Parquet 缓存。

    返回 MultiIndex DataFrame，index=(date, code)，
    columns=[open, high, low, close, volume, amount, turnover]。

    Parameters
    ----------
    codes:
        股票代码列表，6 位数字代码，如 ["000001", "600000"]。
    start_date / end_date:
        日期字符串，格式 "YYYY-MM-DD"。
    cache_dir:
        Parquet 缓存目录，默认 data/cache/。
    request_interval:
        akshare 请求间隔（秒），避免触发限速。
    """
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(codes, start_date, end_date, cache_dir)

    if cache_file.exists():
        logger.info("缓存命中：%s", cache_file.name)
        df = pd.read_parquet(cache_file)
        return df

    logger.info("缓存未命中，从 akshare 拉取 %d 支股票...", len(codes))
    frames: list[pd.DataFrame] = []
    for i, code in enumerate(codes):
        try:
            df = _fetch_single_stock(code, start_date, end_date)
            frames.append(df)
        except Exception as e:
            logger.warning("拉取 %s 失败：%s", code, e)
        if i < len(codes) - 1:
            time.sleep(request_interval)
        if (i + 1) % 50 == 0:
            logger.info("已拉取 %d / %d", i + 1, len(codes))

    if not frames:
        raise RuntimeError("所有股票拉取失败，请检查网络或 akshare 版本。")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index(["date", "code"]).sort_index()
    result.to_parquet(cache_file)
    logger.info("已缓存至 %s", cache_file)
    return result


def load_universe(name: str, date: str | None = None) -> list[str]:
    """获取指数成分股列表。

    Parameters
    ----------
    name:
        股票池名称，支持 "hs300" / "zz500" / "sz50"。
    date:
        查询日期（"YYYY-MM-DD"），None 表示最新。
    """
    import akshare as ak

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
    """获取成分股列表，带简单 TTL 缓存（默认 7 天刷新一次）。"""
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
