"""RL 因子挖掘·训练循环：REINFORCE（优势归一化 + 熵奖励）训练策略挖高 IC 因子。

奖励算在同 regime 训练段（2021+，regime 断点教训）。训练完对 top 因子补算 test_ic，
存成 factor_library 同构 schema，可直接喂 screen_factors.py / deflated 盖章。
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series
from src.gp.evaluator import evaluate, to_wide
from src.rl.env import FactorEnv
from src.rl.policy import FactorPolicy, rollout, sequence_logprob_entropy
from src.rl.tokens import rpn_to_node
from src.utils.logger import setup_experiment_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.01, help="熵奖励系数，防早熟")
    ap.add_argument("--train-start", default="2021-01-01")
    ap.add_argument("--train-end", default="2023-01-01")  # 训练段限在 regime 内
    ap.add_argument("--topk", type=int, default=50, help="落盘并补算 test_ic 的因子数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]

    logger, log_dir = setup_experiment_logger(tag="rl")
    logger.info("设备=%s | 训练段=[%s, %s)", args.device, args.train_start, args.train_end)

    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = to_wide(prices=prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(df=prices, col="fwd_ret")
    lo, hi = pd.Timestamp(args.train_start), pd.Timestamp(args.train_end)

    env = FactorEnv(
        wide=wide, fwd=fwd, train_lo=lo, train_hi=hi, method=method, terminals=terminals
    )
    policy = FactorPolicy(vocab_size=env.vocab_size).to(device=args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    best_reward: dict[str, float] = {}  # expr -> 最佳 |IC|
    best_actions: dict[str, list[int]] = {}  # expr -> token 序列（重建树用）

    for it in range(1, args.iters + 1):
        eps = [rollout(policy=policy, env=env, device=args.device) for _ in range(args.batch)]
        rewards = np.array([e["reward"] for e in eps])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # 优势归一化

        opt.zero_grad()
        loss = torch.zeros((), device=args.device)
        for e, a in zip(eps, adv, strict=True):
            lp, ent = sequence_logprob_entropy(
                policy=policy, actions=e["actions"], masks=e["masks"], device=args.device
            )
            loss = loss - a * lp - args.entropy * ent  # REINFORCE + 熵奖励
        (loss / len(eps)).backward()
        opt.step()

        for e in eps:  # 记历史最优（去重）
            expr = e["info"].get("expr")
            if expr and e["reward"] > best_reward.get(expr, 0.0):
                best_reward[expr] = e["reward"]
                best_actions[expr] = e["actions"]

        if it == 1 or it % 10 == 0:
            top_expr, top_r = max(best_reward.items(), key=lambda kv: kv[1])
            logger.info(
                "iter %3d | 平均|IC|=%.4f 最大|IC|=%.4f 去重因子=%d | 历史最佳 %.4f  %s",
                it,
                rewards.mean(),
                rewards.max(),
                len(best_reward),
                top_r,
                top_expr,
            )

    # ---- 闭环：top 因子补算 test_ic，存成 factor_library 同构 schema ----
    test_lo, test_hi = hi, fwd.index.max() + pd.Timedelta(days=1)
    fwd_te = fwd.loc[(fwd.index >= test_lo) & (fwd.index < test_hi)]
    ranked = sorted(best_reward, key=best_reward.get, reverse=True)[: args.topk]  # type:ignore

    records = []
    for expr in ranked:
        tree = rpn_to_node([env.vocab[a] for a in best_actions[expr]])
        ic_tr = calc_ic_series(
            factor_df=evaluate(node=tree, wide=wide), return_df=env.fwd_train, method=method
        ).dropna()
        ic_te = calc_ic_series(
            factor_df=evaluate(node=tree, wide=wide), return_df=fwd_te, method=method
        ).dropna()

        records.append(
            {
                "expr": expr,
                "tree": tree,
                "size": tree.size(),
                "train_ic": float(ic_tr.mean()),
                "test_ic": float(ic_te.mean()),
            }
        )

    with open(log_dir / "rl_factors.pkl", "wb") as fh:
        pickle.dump({"config_path": args.config, "split": args.train_end, "factors": records}, fh)
    pd.DataFrame(
        [{k: r[k] for k in ("expr", "size", "train_ic", "test_ic")} for r in records]
    ).to_csv(log_dir / "rl_factors.csv", index=False)

    if len(records) >= 3:  # 顺手报训练→OOS 迁移（regime 成败的关键数
        tr = [r["train_ic"] for r in records]
        te = [r["test_ic"] for r in records]
        logger.info(
            "top%d 因子 train_ic vs test_ic 秩相关 = %+.3f",
            len(records),
            spearmanr(tr, te).correlation,
        )
    logger.info("RL 因子落盘：%s（%d 个）", log_dir / "rl_factors.pkl", len(records))


if __name__ == "__main__":
    main()
