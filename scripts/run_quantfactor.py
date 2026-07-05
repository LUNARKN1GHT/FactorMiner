"""QuantFactor REINFORCE (arXiv:2409.05144) 复现·训练循环。

跟 `train_rl.py` 的单因子 REINFORCE 不同，这篇论文的两个核心改动都在这里落地：

1. **贪婪基线**（论文式 10）：每个 episode 跑两条轨迹——一条随机采样、一条贪婪解码
   （`greedy_rollout`），用贪婪轨迹的 reward 当 baseline，不再用 batch 内 reward
   归一化（`train_rl.py` 的做法），省掉了 value network。
2. **因子池 + IR 塑形奖励**（式 13/14）：奖励看的是候选因子加入 `AlphaPool` 后
   组合表现，而非单因子自身 IC，见 `src/rl/pool_env.py`。

同数据、同 tokens/env 语法层、同 `FactorPolicy` 架构，只换算法——跟自家 RL baseline
（`train_rl.py`）保持"唯一变量"，方便横向对比。超参默认从 config 的 `quantfactor:`
段读，命令行可覆盖。
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
from src.rl.alpha_pool import AlphaPool
from src.rl.policy import FactorPolicy, greedy_rollout, rollout, sequence_logprob_entropy
from src.rl.pool_env import PoolFactorEnv
from src.rl.tokens import rpn_to_node
from src.utils.logger import setup_experiment_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--entropy", type=float, help="熵奖励系数，防早熟")
    ap.add_argument("--pool-capacity", type=int)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--topk", type=int, help="落盘因子数上限；0=全部探索到的都存")
    ap.add_argument("--n-jobs", type=int, help="落盘补算 test_ic 的并行核数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    qf = cfg.get("quantfactor", {})

    def pick(cli, key, default):  # 命令行 > config 的 quantfactor: 段 > 兜底
        return cli if cli is not None else qf.get(key, default)

    iters = pick(args.iters, "iters", 300)
    batch = pick(args.batch, "batch", 32)
    lr = pick(args.lr, "lr", 1e-3)
    entropy = pick(args.entropy, "entropy", 0.01)
    pool_capacity = pick(args.pool_capacity, "pool_capacity", 10)
    train_start = pick(args.train_start, "train_start", "2021-01-01")
    train_end = pick(args.train_end, "train_end", "2023-01-01")
    topk = pick(args.topk, "topk", 0)
    n_jobs = pick(args.n_jobs, "n_jobs", 10)
    max_len = qf.get("max_len", 10)
    pool_lr = qf.get("pool_lr", 0.05)
    pool_gd_steps = qf.get("pool_gd_steps", 200)
    ir_alpha = qf.get("ir_alpha", 180_000.0)
    ir_eta = qf.get("ir_eta", 1e-6)
    ir_delta = qf.get("ir_delta", 0.3)
    ir_lambda = qf.get("ir_lambda", 0.02)
    seed = qf.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger, log_dir = setup_experiment_logger(tag="quantfactor")
    logger.info(
        "设备=%s seed=%d | 训练段=[%s, %s) | iters=%d batch=%d lr=%g entropy=%g "
        "pool_capacity=%d max_len=%d",
        args.device, seed, train_start, train_end, iters, batch, lr, entropy,
        pool_capacity, max_len,
    )
    logger.info(
        "IR 塑形超参：alpha=%g eta=%g delta=%g lambda=%g | 池子拟合：lr=%g gd_steps=%d",
        ir_alpha, ir_eta, ir_delta, ir_lambda, pool_lr, pool_gd_steps,
    )

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")
    lo, hi = pd.Timestamp(train_start), pd.Timestamp(train_end)
    fwd_train = fwd.loc[(fwd.index >= lo) & (fwd.index < hi)]

    pool = AlphaPool(
        fwd=fwd_train, capacity=pool_capacity, method=method, lr=pool_lr, gd_steps=pool_gd_steps,
        seed=seed,
    )
    env = PoolFactorEnv(
        wide=wide, fwd=fwd, train_lo=lo, train_hi=hi, pool=pool, method=method,
        terminals=terminals, max_len=max_len,
        ir_alpha=ir_alpha, ir_eta=ir_eta, ir_delta=ir_delta, ir_lambda=ir_lambda,
    )
    policy = FactorPolicy(vocab_size=env.vocab_size).to(device=args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    best_reward: dict[str, float] = {}  # expr -> 最佳 reward（用于落盘取 topk 时排序）
    best_actions: dict[str, list[int]] = {}  # expr -> token 序列（重建树用）
    global_step = 0

    for it in range(1, iters + 1):
        sampled_eps, greedy_eps = [], []
        for _ in range(batch):
            env.global_step = global_step
            sampled = rollout(policy=policy, env=env, device=args.device)
            env.global_step = global_step
            greedy = greedy_rollout(policy=policy, env=env, device=args.device)
            sampled_eps.append(sampled)
            greedy_eps.append(greedy)
            global_step += 1

        # 贪婪基线 REINFORCE（论文式 10）：advantage = r(采样轨迹) − r(贪婪轨迹)，
        # 不做 batch 内 reward 归一化——这正是这篇论文相对 train_rl.py 的核心差异。
        opt.zero_grad()
        loss = torch.zeros((), device=args.device)
        for sampled, greedy in zip(sampled_eps, greedy_eps, strict=True):
            lp, ent = sequence_logprob_entropy(
                policy=policy, actions=sampled["actions"], masks=sampled["masks"],
                device=args.device,
            )
            advantage = sampled["reward"] - greedy["reward"]
            loss = loss - advantage * lp - entropy * ent
        (loss / batch).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()

        for ep in (*sampled_eps, *greedy_eps):  # 两条轨迹都是合法探索到的候选因子
            info = ep["info"]
            expr = info.get("expr")
            if info.get("ic") is not None and ep["reward"] > best_reward.get(expr, float("-inf")):
                best_reward[expr] = ep["reward"]
                best_actions[expr] = ep["actions"]

        if it == 1 or it % 10 == 0:
            sampled_r = np.array([e["reward"] for e in sampled_eps])
            top_expr, top_r = max(best_reward.items(), key=lambda kv: kv[1], default=(None, 0.0))
            logger.info(
                "iter %3d | 采样轨迹平均 reward=%.4f 最大=%.4f 去重因子=%d | 历史最佳 %.4f  %s",
                it, sampled_r.mean(), sampled_r.max(), len(best_reward), top_r, top_expr,
            )

    # ---- 落盘探索到的全部因子：每个因子自身的 train/test IC（不是池组合分数），
    # 跟 GP/RL/LLM 的 factor_library 口径一致，方便横向对比 ----
    test_hi = fwd.index.max() + pd.Timedelta(days=1)
    exprs = sorted(best_reward, key=best_reward.get, reverse=True)  # type: ignore[arg-type]
    if topk > 0:
        exprs = exprs[:topk]

    vocab = env.vocab

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

    with open(log_dir / "factor_library.pkl", "wb") as fh:
        pickle.dump(
            {
                "config_path": args.config,
                "split": train_end,
                "n_explored": len(best_reward),
                "factors": records,
            },
            fh,
        )
    pd.DataFrame(
        [{k: r[k] for k in ("expr", "size", "train_ic", "test_ic")} for r in records]
    ).to_csv(log_dir / "quantfactor_factors.csv", index=False)

    if len(records) >= 3:
        tr = [r["train_ic"] for r in records]
        te = [r["test_ic"] for r in records]
        logger.info("全库 train_ic vs test_ic 秩相关 = %+.3f", spearmanr(tr, te).correlation)
    logger.info(
        "QuantFactor 因子落盘：%s（存 %d / 探索 %d 个）",
        log_dir / "factor_library.pkl", len(records), len(best_reward),
    )


if __name__ == "__main__":
    main()
