"""数据清洗脚本：读取缓存行情，清洗后保存为 prices_clean.parquet。

用法：
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --universe hs300 --start 2018-01-01 --end 2024-12-31 --period 5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.loader import load_daily_prices, load_universe_cached
from data.preprocess import clean_prices, compute_returns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗行情数据并保存")
    parser.add_argument("--universe", default="hs300")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--period", type=int, default=5, help="forward return 窗口（交易日）")
    parser.add_argument("--out", default="data/cache/prices_clean.parquet", help="输出路径")
    args = parser.parse_args()

    codes = load_universe_cached(args.universe)
    raw = load_daily_prices(codes, args.start, args.end)

    prices = clean_prices(raw)
    prices = compute_returns(prices, period=args.period)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    print(f"已保存：{out}  shape={prices.shape}")
    print(f"列：{list(prices.columns)}")


if __name__ == "__main__":
    main()
