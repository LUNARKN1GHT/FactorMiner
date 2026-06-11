"""因子动物园·生成器：大批量生成因子，算 train/test mean IC，存成因子库（只生成不筛选）。

生成与筛选解耦——这里只管广撒网 + 存档，筛选交给 screen_factors.py。
指标用 mean rank IC（mentor：用 IC 不用 ICIR）。
"""

import argparse
import pickle
from collections.abc import Callable
from typing import TypedDict

import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.core.evaluator import evaluate, to_wide
from src.core.tree import Node
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.gp.engine import ramped_half_and_half
from src.gp.stgp import typed_ramped_half_and_half
from src.utils.logger import setup_experiment_logger

MIN_DISTINCT = 4  # 退化守卫：每日截面去重中位数下限，挡近常数因子


class FactorRecord(TypedDict):
    """因子库里的一条记录——每个字段类型明确"""

    expr: str  # 表达式字符串（去重键 / 人读）
    tree: Node  # 表达式树本体（可重新求值）
    size: int  # 树节点数
    train_ic: float  # 训练段 mean rank IC（主指标）
    test_ic: float  # 测试段 mean rank IC
    train_icir: float  # 训练段 ICIR，仅参考、不用于筛选


def generate_unique_trees(
    n_target: int, min_depth: int, max_depth: int, strongly_typed: bool, max_tries: int = 50
) -> list[Node]:
    """反复 ramped 生成并按 str(tree) 去重，攒够 n_target 棵不同的树。"""
    gen: Callable[[int], list[Node]]
    if strongly_typed:
        gen = lambda k: typed_ramped_half_and_half(k, min_depth, max_depth)  # noqa: E731
    else:
        gen = lambda k: ramped_half_and_half(k, min_depth, max_depth)  # noqa: E731
    seen: dict[str, object] = {}
    for _ in range(max_tries):
        if len(seen) >= n_target:
            break
        for t in gen(n_target):
            seen.setdefault(str(t), t)
    return list(seen.values())[:n_target]  # type: ignore


def eval_one(
    tree: Node,
    wide: dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    method: str,
    train_lo: pd.Timestamp,
    split: pd.Timestamp,
    test_hi: pd.Timestamp,
) -> FactorRecord | None:
    """算一棵树的 train/test 平均 IC；退化/算崩返回 None。"""
    try:
        fac = evaluate(tree, wide)
    except Exception:
        return None
    if fac.nunique(axis=1).median() < MIN_DISTINCT:  # 退化守卫
        return None
    try:
        ic = calc_ic_series(fac, fwd, method=method)
    except Exception:
        return None
    ic_tr = ic.loc[(ic.index >= train_lo) & (ic.index < split)].dropna()
    ic_te = ic.loc[(ic.index >= split) & (ic.index < test_hi)].dropna()
    if len(ic_tr) < 20 or len(ic_te) < 20:  # 两段都得有足够观测
        return None
    return {
        "expr": str(tree),
        "tree": tree,
        "size": tree.size(),
        "train_ic": float(ic_tr.mean()),  # ← 主指标：mean rank IC
        "test_ic": float(ic_te.mean()),
        "train_icir": float(ic_tr.mean() / ic_tr.std()) if ic_tr.std() > 0 else 0.0,  # 仅参考
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=5000, help="目标生成因子数")
    ap.add_argument("--split", default="2023-01-01", help="train/test 分界")
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument(
        "--train-start",
        default=None,
        help="训练段起点（默认数据最早；设 2021-01-01 可避开 2020 regime 断点）",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    gp = cfg.get("gp", {})
    min_d, max_d = gp.get("min_depth", 2), gp.get("max_depth", 6)
    typed = gp.get("strongly_typed", False)

    logger, log_dir = setup_experiment_logger(tag="factorzoo")

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")
    train_lo = pd.Timestamp(args.train_start) if args.train_start else fwd.index.min()
    split = pd.Timestamp(args.split)
    test_hi = fwd.index.max() + pd.Timedelta(days=1)

    logger.info("生成 %d 棵去重因子树 ...", args.n)
    trees = generate_unique_trees(args.n, min_d, max_d, typed)
    logger.info("去重后 %d 棵，并行求值 IC（n_jobs=%d）...", len(trees), args.n_jobs)

    results = Parallel(n_jobs=args.n_jobs, backend="loky", batch_size=32)(
        delayed(eval_one)(t, wide, fwd, method, train_lo, split, test_hi) for t in trees
    )
    library = [r for r in results if r is not None]
    logger.info("有效因子 %d / %d（其余退化/算崩被丢）", len(library), len(trees))

    log_dir.mkdir(parents=True, exist_ok=True)  # 防御性，几小时的活别丢
    with open(log_dir / "factor_library.pkl", "wb") as fh:
        pickle.dump({"config_path": args.config, "split": str(split), "factors": library}, fh)
    # 顺手存标量 CSV，不开 python 也能 sort 看
    pd.DataFrame(
        [{k: r[k] for k in ("expr", "size", "train_ic", "test_ic", "train_icir")} for r in library]
    ).to_csv(log_dir / "factor_library.csv", index=False)
    logger.info("因子库已落盘：%s", log_dir / "factor_library.pkl")


if __name__ == "__main__":
    main()
