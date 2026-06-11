"""数据预拉取脚本：下载并缓存指定股票池的历史行情。

用法:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --universe hs300 --start 2020-01-01 --end 2023-12-31
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.loader import load_daily_prices, load_universe_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="预拉取 A 股行情数据")
    parser.add_argument("--universe", default="hs300", help="股票池 (hs300/zz500/sz50)")
    parser.add_argument("--start", default="2018-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--interval", type=float, default=None, help="请求间隔（秒），默认由数据源决定"
    )
    args = parser.parse_args()

    print(f"获取 {args.universe} 成分股...")
    codes = load_universe_cached(args.universe)
    print(f"共 {len(codes)} 只，示例：{codes[:5]}")

    print(f"拉取行情 {args.start} ~ {args.end}（首次运行较慢）...")
    df = load_daily_prices(codes, args.start, args.end, request_interval=args.interval)

    print("\n数据概况：")
    print(f"  shape: {df.shape}")
    print(
        f"  日期范围: {df.index.get_level_values('date').min()} ~ {df.index.get_level_values('date').max()}"
    )
    print(f"  股票数量: {df.index.get_level_values('code').nunique()}")
    print(f"  列: {list(df.columns)}")
    print("\n前5行：")
    print(df.head())


if __name__ == "__main__":
    main()
