"""AlphaGen 复现·桥接入口：把官方仓库跑出的因子池 json 翻译回本项目 Node，

在自己的 tushare/hs300 数据上用自己的评估流水线重新算 train/test IC，落盘成
factor_library 同构 schema，喂 `compare_generators.py` 跟 GP/RL/LLM/QuantFactor
横向对比。翻译逻辑见 `src/rl/alphagen_bridge.py`（算子映射表 + 不可翻译判定）。

用法：
    python scripts/bridge_alphagen.py <pool.json> [config]

<pool.json> 是官方仓库 `CustomCallback.save_checkpoint` 落的
`{step}_steps_pool.json`，schema 为 ``{"exprs": [...], "weights": [...]}``
（见 alphagen/models/linear_alpha_pool.py::to_json_dict）。
"""

from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd
import yaml

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.rl.alphagen_bridge import translate_pool
from src.utils.logger import setup_experiment_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_json", help="官方仓库落盘的 {step}_steps_pool.json 路径")
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--train-end", help="train/test 切分点；默认用 config 的 rl.train_end")
    args = ap.parse_args()

    with open(args.pool_json, encoding="utf-8") as f:
        pool = json.load(f)
    exprs: list[str] = pool["exprs"]

    ok, failed = translate_pool(exprs)
    logger, log_dir = setup_experiment_logger(tag="alphagen")
    logger.info(
        "因子池共 %d 条，可翻译 %d 条（%.0f%%），跳过 %d 条",
        len(exprs), len(ok), 100 * len(ok) / max(len(exprs), 1), len(failed),
    )
    for expr, reason in failed:
        logger.info("跳过：%s | 原因：%s", expr, reason)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    train_end = args.train_end or cfg.get("rl", {}).get("train_end", "2023-01-01")

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")
    hi = pd.Timestamp(train_end)
    test_hi = fwd.index.max() + pd.Timedelta(days=1)

    records = []
    for expr, tree in ok:
        try:
            ic = calc_ic_series(factor_df=evaluate(node=tree, wide=wide), return_df=fwd, method=method)
        except Exception as e:
            logger.info("桥接后求值失败：%s | %s", expr, e)
            continue
        ic_tr = ic.loc[ic.index < hi].dropna()
        ic_te = ic.loc[(ic.index >= hi) & (ic.index < test_hi)].dropna()
        records.append({
            "expr": expr,
            "tree": tree,
            "size": tree.size(),
            "train_ic": float(ic_tr.mean()) if len(ic_tr) else float("nan"),
            "test_ic": float(ic_te.mean()) if len(ic_te) else float("nan"),
        })

    with open(log_dir / "factor_library.pkl", "wb") as fh:
        pickle.dump(
            {
                "config_path": args.config,
                "split": str(train_end),
                "n_explored": len(exprs),  # 官方池子的真实候选数（含翻译失败的）
                "factors": records,
            },
            fh,
        )
    logger.info(
        "AlphaGen 桥接落盘：%s（%d 条在自有数据上重新求值成功）",
        log_dir / "factor_library.pkl", len(records),
    )


if __name__ == "__main__":
    main()
