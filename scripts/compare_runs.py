"""跨 run 对照：把多个 walkforward.pkl（普通 GP vs STGP）的样本外战绩并排比。

只读 pkl，不需要数据。每个 run 取「每折按 train|ICIR| 选出的因子」的 OOS 表现，
报逐折 test|ICIR|、均值/标准差/最差、符号翻转折数，外加前沿含 rank 因子占比与平均树大小。
"""

import pickle
import statistics
import sys
from pathlib import Path


def summarize_run(payload: dict) -> dict:
    folds = payload["folds"]
    picks = payload["picks"]  # (fold, size, |tr|, |te|, decay, tree, 带符号tr, 带符号te)

    oos = [float(p[3]) for p in picks]  # 每折选中因子的 test|ICIR|
    flips = sum(1 for p in picks if p[6] * p[7] < 0)  # 训练/测试符号相反 = 方向翻转

    rows = [r for f in folds for r in f["rows"]]  # 所有折的整条前沿
    n_rank = sum(1 for r in rows if "rank(" in r["expr"])  # 用字符串判，跨 run 可比
    return {
        "oos": oos,
        "mean": statistics.mean(oos),
        "std": statistics.pstdev(oos),
        "worst": min(oos),
        "flips": flips,
        "n_folds": len(picks),
        "rank_frac": n_rank / len(rows) if rows else 0.0,
        "avg_size": statistics.mean(r["size"] for r in rows) if rows else 0.0,
    }


def main(paths: list[str]) -> None:
    # 用 exp 目录名当 run 标签（results/logs/exp_xxx/walkforward.pkl → exp_xxx）
    runs = []
    for path in paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        runs.append((Path(path).parent.name, summarize_run(payload)))

    print("=== 逐折 OOS |ICIR| ===")
    print(f"{'run':>22} {'各折':>24} {'均值':>8} {'标准差':>8} {'最差':>8} {'翻转':>6}")
    for label, s in runs:
        folds_str = " ".join(f"{x:.3f}" for x in s["oos"])
        print(
            f"{label:>22} {folds_str:>24} {s['mean']:>8.4f} "
            f"{s['std']:>8.4f} {s['worst']:>8.4f} {s['flips']:>4}/{s['n_folds']}"
        )

    print("\n=== 前沿结构 ===")
    print(f"{'run':>22} {'含rank因子占比':>14} {'平均树大小':>12}")
    for label, s in runs:
        print(f"{label:>22} {s['rank_frac']:>13.1%} {s['avg_size']:>12.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("用法：python scripts/compare_runs.py <pkl1> <pkl2> [更多...]")
    main(sys.argv[1:])
