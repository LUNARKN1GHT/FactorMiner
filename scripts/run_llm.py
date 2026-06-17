"""LLM 进化挖因子·主循环（操作员 A + 评论家 B）。详见 doc/LLM进化挖因子.md。

每轮：A 生成 → clean（砍复杂/跨轮去重）→ 只对新因子在 TRAIN 段求值（NaN/常数=退化，丢）
→ 更新 archive（按 train_score=|train_ic|−parsimony·size，只看 train）→ B 诊断 → 回灌下一轮。
早停后对全部 unique 因子算一次 test_ic（仅此一次，永不回流），存 factor_library 同构 schema。

铁律：搜索回路只看 train，test_ic 永不进回路。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle

import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.llm.clean import clean, to_expr
from src.llm.critic import critique
from src.llm.generate import call_llm, parse_response
from src.llm.parser import CORE_TERMINALS
from src.llm.prompts import load_prompt
from src.llm.usage import Usage
from src.utils.logger import log_generation, setup_experiment_logger


def _fmt(rows: list[dict]) -> str:
    """因子列表 → 纯文本（带窗口的 expr + train 指标，不含 test_ic）。"""
    return (
        "\n".join(f"{r['expr']} | train_ic={r['train_ic']:.4f} size={r['size']}" for r in rows)
        or "（暂无）"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--n-per-round", type=int)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    llm = cfg.get("llm", {})

    def pick(cli, key, default):
        """命令行 > config 的 llm"""
        return cli if cli is not None else llm.get(key, default)

    # 程序配置读取
    rounds = pick(args.rounds, "rounds", 10)
    n_per_round = pick(args.n_per_round, "n_per_round", 50)
    patience = llm.get("patience", 3)
    critic_every = llm.get("critic_every", 1)
    temperature = llm.get("temperature", 1.0)
    max_operators = llm.get("max_operators", 10)
    parsimony = llm.get("parsimony", 0.001)
    archive_size = llm.get("archive_size", 10)
    train_start = llm.get("train_start", "2021-01-01")
    train_end = llm.get("train_end", "2023-01-01")
    topk = llm.get("topk", 0)
    n_jobs = llm.get("n_jobs", 10)
    model_a = llm.get("model_a", "deepseek-v4-pro")
    model_b = llm.get("model_b", "deepseek-v4-pro")
    price_in = llm.get("price_in", 0.0)  # 每百万 token 单价（估算金额用）
    price_out = llm.get("price_out", 0.0)

    logger, log_dir = setup_experiment_logger(tag="llm")
    logger.info(
        "训练段=[%s, %s) | rounds=%d n/round=%d patience=%d parsimony=%g max_op=%d",
        train_start,
        train_end,
        rounds,
        n_per_round,
        patience,
        parsimony,
        max_operators,
    )

    # 获取价格列表，并截取出训练段
    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")
    lo, hi = pd.Timestamp(train_start), pd.Timestamp(train_end)
    test_hi = fwd.index.max() + pd.Timedelta(days=1)

    def train_ic_of(tree) -> float:
        """只算训练段 IC；求值失败/退化 -> NaN"""
        try:
            ic = calc_ic_series(
                factor_df=evaluate(node=tree, wide=wide), return_df=fwd, method=method
            )
        except Exception:
            return float("nan")
        ic_tr = ic.loc[(ic.index >= lo) & (ic.index < hi)].dropna()
        return float(ic_tr.mean()) if len(ic_tr) else float("nan")

    def score(rec: dict) -> float:
        """训练段因子的得分"""
        return abs(rec["train_ic"]) - parsimony * rec["size"]

    seen: set[str] = set()  # 全局 canon_key
    archive: list[dict] = []  # 跨轮精英
    explored: list[dict] = []  # 所有有效 unique 因子
    avoid: list[str] = []  # 退化因子 canon_ke
    critique_text: str = ""
    best: float = float("-inf")  # 最好的 ic
    no_improve: int = 0  # 没有进化的轮数
    usage = Usage()  # 累计本次实验的 token / 金额

    for r in range(1, rounds + 1):
        rd = log_dir / "rounds" / f"round_{r:02d}"
        rd.mkdir(parents=True, exist_ok=True)

        if r == 1:
            # 第一轮，不需要评论员处理
            prompt = load_prompt(
                "factor_generation", n=n_per_round, terminals=", ".join(CORE_TERMINALS)
            )
        else:
            prompt = load_prompt(
                "factor_evolution",
                n=n_per_round,
                terminals=", ".join(CORE_TERMINALS),
                archive=_fmt(archive),
                avoid="\n".join(avoid[-30:]) or "（暂无）",
                critique=critique_text or "（暂无）",
            )
        text = call_llm(prompt, model=model_a, temperature=temperature, usage=usage)
        (rd / "prompt_a.txt").write_text(prompt, encoding="utf-8")
        (rd / "response_a.txt").write_text(text, encoding="utf-8")

        new = clean(
            parse_response(text, CORE_TERMINALS, logger=logger),
            seen=seen,
            max_operators=max_operators,
        )

        results: list[dict] = []
        for t in new:
            ic = train_ic_of(t)
            if math.isnan(ic):
                avoid.append(to_expr(t))
                continue
            results.append({"expr": to_expr(t), "tree": t, "size": t.size(), "train_ic": ic})
        explored.extend(results)

        archive = sorted(archive + results, key=score, reverse=True)[:archive_size]

        if results and r % critic_every == 0:
            # 模型 B 诊断
            critique_text = critique(results, archive, model=model_b, logger=logger, usage=usage)
            (rd / "response_b.txt").write_text(critique_text, encoding="utf-8")

        round_best = max((score(x) for x in results), default=float("-inf"))
        mean_ic = sum(x["train_ic"] for x in results) / len(results) if results else float("nan")
        log_generation(
            log_dir,
            gen=r,
            best_ic=round_best if results else float("nan"),
            mean_ic=mean_ic,
            best_expr=archive[0]["expr"] if archive else "",
            extra={
                "n_new": len(new),
                "n_valid": len(results),
                "n_explored": len(seen),
                "n_archive": len(archive),
            },
        )
        logger.info(
            "round %d | 新=%d 有效%d 探索累计=%d | 最佳 train_score=%.4f",
            r,
            len(new),
            len(results),
            len(seen),
            round_best,
        )

        if round_best > best + 1e-6:
            best, no_improve = round_best, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info("连续 %d 轮无提升，早停于 round %d", patience, r)
            break

    # --- 全部 unique 因子补算 test_ic ---
    if topk > 0:
        explored = sorted(explored, key=score, reverse=True)[:topk]

    def with_test(rec: dict) -> dict:
        ic = calc_ic_series(
            factor_df=evaluate(node=rec["tree"], wide=wide), return_df=fwd, method=method
        )
        ic_te = ic.loc[(ic.index >= hi) & (ic.index < test_hi)].dropna()
        return {**rec, "test_ic": float(ic_te.mean()) if len(ic_te) else float("nan")}

    records = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16)(
        delayed(with_test)(rec) for rec in explored
    )

    with open(log_dir / "factor_library.pkl", "wb") as fh:
        pickle.dump(
            {
                "config_path": args.config,
                "split": train_end,
                "n_explored": len(seen),
                "factors": records,
            },
            fh,
        )
    pd.DataFrame(
        [{k: r[k] for k in ("expr", "size", "train_ic", "test_ic")} for r in records]
    ).to_csv(log_dir / "factor_library.csv", index=False)
    logger.info(
        "LLM 因子落盘：%s（存 %d / 探索 %d）",
        log_dir / "factor_library.pkl",
        len(records),
        len(seen),
    )

    with open(log_dir / "usage.json", "w", encoding="utf-8") as uf:
        json.dump(usage.as_dict(price_in, price_out), uf, ensure_ascii=False, indent=2)
    logger.info(
        "本次实验 LLM 用量：%d 次调用 | %d tokens（in %d / out %d）| 估算 ¥%.4f",
        usage.calls,
        usage.total_tokens,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cost(price_in, price_out),
    )


if __name__ == "__main__":
    main()
