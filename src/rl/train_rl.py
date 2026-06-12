"""RL 因子挖掘·训练循环：REINFORCE（优势归一化 + 熵奖励）训练策略挖高 IC 因子。

奖励算在同 regime 训练段（2021+，regime 断点教训）。训练完把**探索到的全部因子**补算 test_ic，
存成 factor_library 同构 schema（含 n_explored=真实试验数，供 deflated 用对的 N），
可直接喂 screen_factors.py 盖章。超参默认从 config 的 `rl:` 段读，命令行可覆盖。
"""

from __future__ import annotations

import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.rl.env import FactorEnv
from src.rl.policy import FactorPolicy, rollout, sequence_logprob_entropy
from src.rl.tokens import rpn_to_node
from src.utils.logger import setup_experiment_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    # 以下默认 None：None 时用 config 的 rl: 段，命令行给了就覆盖
    ap.add_argument("--iters", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--entropy", type=float, help="熵奖励系数，防早熟")
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")  # 训练段限在 regime 内
    ap.add_argument("--topk", type=int, help="落盘因子数上限；0=全部探索到的都存")
    ap.add_argument("--n-jobs", type=int, help="落盘补算 test_ic 的并行核数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    rl = cfg.get("rl", {})

    def pick(cli, key, default):  # 命令行 > config 的 rl: 段 > 兜底
        return cli if cli is not None else rl.get(key, default)

    iters = pick(args.iters, "iters", 300)
    batch = pick(args.batch, "batch", 64)
    lr = pick(args.lr, "lr", 1e-3)
    entropy = pick(args.entropy, "entropy", 0.01)
    train_start = pick(args.train_start, "train_start", "2021-01-01")
    train_end = pick(args.train_end, "train_end", "2023-01-01")
    topk = pick(args.topk, "topk", 0)
    n_jobs = pick(args.n_jobs, "n_jobs", 10)
    parsimony = rl.get("parsimony", 0.0)  # 简约惩罚系数（罚 bloat 过拟合）
    max_len = rl.get("max_len", 20)  # 表达式最大 token 数（硬卡树大小）
    seed = rl.get("seed", 42)  # 固定种子保可复现（baseline 必须）

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # CPU 时无副作用

    logger, log_dir = setup_experiment_logger(tag="rl")
    logger.info(
        "设备=%s seed=%d | 训练段=[%s, %s) | iters=%d batch=%d lr=%g entropy=%g parsimony=%g max_len=%d",
        args.device, seed, train_start, train_end, iters, batch, lr, entropy, parsimony, max_len,
    )

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")
    lo, hi = pd.Timestamp(train_start), pd.Timestamp(train_end)

    env = FactorEnv(
        wide=wide, fwd=fwd, train_lo=lo, train_hi=hi, method=method, terminals=terminals,
        max_len=max_len, parsimony=parsimony,
    )
    policy = FactorPolicy(vocab_size=env.vocab_size).to(device=args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    best_reward: dict[str, float] = {}  # expr -> 最佳 |IC|
    best_actions: dict[str, list[int]] = {}  # expr -> token 序列（重建树用）

    for it in range(1, iters + 1):
        eps = [rollout(policy=policy, env=env, device=args.device) for _ in range(batch)]
        rewards = np.array([e["reward"] for e in eps])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # 优势归一化

        opt.zero_grad()
        loss = torch.zeros((), device=args.device)
        for e, a in zip(eps, adv, strict=True):
            lp, ent = sequence_logprob_entropy(
                policy=policy, actions=e["actions"], masks=e["masks"], device=args.device
            )
            loss = loss - a * lp - entropy * ent  # REINFORCE + 熵奖励
        (loss / len(eps)).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()

        for e in eps:  # 记历史最优（只记有效因子=算出了 IC 的；奖励含罚则、可为负）
            info = e["info"]
            expr = info.get("expr")
            if info.get("ic") is not None and e["reward"] > best_reward.get(expr, float("-inf")):
                best_reward[expr] = e["reward"]
                best_actions[expr] = e["actions"]

        if it == 1 or it % 10 == 0:
            top_expr, top_r = max(best_reward.items(), key=lambda kv: kv[1])
            logger.info(
                "iter %3d | 平均奖励=%.4f 最大奖励=%.4f 去重因子=%d | 历史最佳 %.4f  %s",
                it, rewards.mean(), rewards.max(), len(best_reward), top_r, top_expr,
            )

    # ---- 落盘探索到的全部因子（一次求值同时算 train/test IC），factor_library 同构 schema ----
    test_hi = fwd.index.max() + pd.Timedelta(days=1)
    exprs = sorted(best_reward, key=best_reward.get, reverse=True)  # type: ignore[arg-type]
    if topk > 0:  # 默认 0 = 全存
        exprs = exprs[:topk]

    vocab = env.vocab  # 局部捕获，避免 joblib pickle 整个 env

    def _record(expr: str, actions: list[int]) -> dict:
        tree = rpn_to_node([vocab[a] for a in actions])
        ic = calc_ic_series(factor_df=evaluate(node=tree, wide=wide), return_df=fwd, method=method)
        ic_tr = ic.loc[(ic.index >= lo) & (ic.index < hi)].dropna()
        ic_te = ic.loc[(ic.index >= hi) & (ic.index < test_hi)].dropna()
        return {
            "expr": expr,
            "tree": tree,
            "size": tree.size(),
            "train_ic": float(ic_tr.mean()) if len(ic_tr) else float("nan"),
            "test_ic": float(ic_te.mean()) if len(ic_te) else float("nan"),
        }

    records = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16)(
        delayed(_record)(e, best_actions[e]) for e in exprs
    )

    with open(log_dir / "rl_factors.pkl", "wb") as fh:
        pickle.dump(
            {
                "config_path": args.config,
                "split": train_end,
                "n_explored": len(best_reward),  # 真实试验数，供 deflated 用对的 N
                "factors": records,
            },
            fh,
        )
    pd.DataFrame(
        [{k: r[k] for k in ("expr", "size", "train_ic", "test_ic")} for r in records]
    ).to_csv(log_dir / "rl_factors.csv", index=False)

    if len(records) >= 3:  # 顺手报训练→OOS 迁移（regime 成败关键数）
        tr = [r["train_ic"] for r in records]
        te = [r["test_ic"] for r in records]
        logger.info("全库 train_ic vs test_ic 秩相关 = %+.3f", spearmanr(tr, te).correlation)
    logger.info(
        "RL 因子落盘：%s（存 %d / 探索 %d 个）",
        log_dir / "rl_factors.pkl", len(records), len(best_reward),
    )


if __name__ == "__main__":
    main()
