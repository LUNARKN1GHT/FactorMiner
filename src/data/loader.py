"""A 股行情数据加载模块，支持 akshare / tushare，带逐股缓存与断点续传。

数据源通过环境变量或参数选择：
  DATA_SOURCE=tushare  # 或 akshare（默认）
  TUSHARE_TOKEN=your_token
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"

_AKSHARE_COL_MAP = {
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

_KEEP_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def _load_env() -> None:
    """尝试加载项目根目录的 .env 文件。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parents[2] / ".env")
    except ImportError:
        pass


def _stock_cache_path(code: str, start_date: str, end_date: str, stock_cache_dir: Path) -> Path:
    return stock_cache_dir / f"{code}_{start_date}_{end_date}.parquet"


# ---------------------------------------------------------------------------
# akshare
# ---------------------------------------------------------------------------


def _fetch_akshare(code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

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
            wait = 2**attempt
            logger.debug(
                "akshare 拉取 %s 第 %d 次失败，%.0fs 后重试：%s", code, attempt + 1, wait, e
            )
            time.sleep(wait)

    df = df.rename(columns=_AKSHARE_COL_MAP)
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    return df


# ---------------------------------------------------------------------------
# tushare
# ---------------------------------------------------------------------------


def _to_ts_code(code: str) -> str:
    """6 位代码 → tushare 格式，如 '000001' → '000001.SZ'。"""
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


def _get_tushare_pro():
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("未找到 TUSHARE_TOKEN，请在 .env 中设置。")
    ts.set_token(token)
    return ts.pro_api()


def _fetch_tushare(
    code: str, start_date: str, end_date: str, pro, retries: int = 3
) -> pd.DataFrame:
    ts_code = _to_ts_code(code)

    for attempt in range(retries):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adj="qfq",
            )
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            logger.debug(
                "tushare 拉取 %s 第 %d 次失败，%.0fs 后重试：%s", code, attempt + 1, wait, e
            )
            time.sleep(wait)

    # tushare 列名：trade_date open high low close vol amount
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["code"] = code
    keep = [c for c in _KEEP_COLS if c in df.columns]
    return df[keep + ["code"]].copy()


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------


def load_daily_prices(
    codes: list[str],
    start_date: str,
    end_date: str,
    source: str | None = None,
    cache_dir: Path | str | None = None,
    request_interval: float | None = None,
) -> pd.DataFrame:
    """加载日频行情数据（前复权），逐股缓存 + 断点续传。

    返回 MultiIndex DataFrame，index=(date, code)，
    columns=[open, high, low, close, volume, amount, ...]。

    Parameters
    ----------
    codes:
        股票代码列表，6 位数字，如 ["000001", "600000"]。
    start_date / end_date:
        日期字符串，格式 "YYYY-MM-DD"。
    source:
        数据源，"akshare" 或 "tushare"。
        None 时读取环境变量 DATA_SOURCE，默认 "akshare"。
    cache_dir:
        缓存根目录，默认 data/cache/。
    request_interval:
        请求间隔（秒）。akshare 默认 0.5s，tushare 默认 0.3s。
    """
    _load_env()

    source = (source or os.environ.get("DATA_SOURCE", "akshare")).lower()  # type: ignore
    if source not in ("akshare", "tushare"):
        raise ValueError(f"不支持的数据源：{source}，可选 akshare / tushare")

    if request_interval is None:
        request_interval = 0.5 if source == "akshare" else 2.0  # tushare 免费版 30次/分钟

    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    stock_cache_dir = cache_dir / "stocks" / source
    stock_cache_dir.mkdir(parents=True, exist_ok=True)

    pro = _get_tushare_pro() if source == "tushare" else None

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for code in codes:
        p = _stock_cache_path(code, start_date, end_date, stock_cache_dir)
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append(code)

    if missing:
        logger.info(
            "[%s] 已缓存 %d 只，待拉取 %d 只", source, len(codes) - len(missing), len(missing)
        )
        for i, code in enumerate(missing):
            try:
                df = (
                    _fetch_akshare(code, start_date, end_date)
                    if source == "akshare"
                    else _fetch_tushare(code, start_date, end_date, pro)
                )
                p = _stock_cache_path(code, start_date, end_date, stock_cache_dir)
                df.to_parquet(p)
                frames.append(df)
            except Exception as e:
                logger.warning("[%s] 拉取 %s 失败（跳过）：%s", source, code, e)
            if i < len(missing) - 1:
                time.sleep(request_interval)
            if (i + 1) % 50 == 0:
                logger.info("进度 %d / %d", len(codes) - len(missing) + i + 1, len(codes))
    else:
        logger.info("[%s] 全部 %d 只命中缓存", source, len(codes))

    if not frames:
        raise RuntimeError("没有任何数据，请检查网络或数据源配置。")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index(["date", "code"]).sort_index()
    return result


def load_universe(name: str, date: str | None = None) -> list[str]:
    """获取指数成分股列表（akshare 实时）。"""
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
