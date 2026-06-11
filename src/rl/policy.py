"""RL 因子挖掘·策略网络：GRU 自回归建模 token 序列，逐步输出下一个 token 的分布。

- FactorPolicy：embed → GRU → 词表 logits
- rollout：用策略 + 环境的 legal_mask 采样生成一条合法因子（无梯度）
- sequence_logprob_entropy：teacher-forcing 重算这条序列的 logprob 与熵（带梯度，供训练）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.rl.env import FactorEnv


class FactorPolicy(nn.Module):
    """逐 token 生成因子的自回归策略。"""

    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bos = vocab_size  # 句首符 index（独立于词表，放在最后一位）
        self.embed = nn.Embedding(vocab_size + 1, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(
        self, inp: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """inp:(B,T) long → logits:(B,T,V) 与新 hidden。"""
        out, h = self.gru(self.embed(inp), h)
        return self.head(out), h  # type: ignore


@torch.no_grad()
def rollout(policy: FactorPolicy, env: FactorEnv, device: str) -> dict:
    """用策略采样生成一条因子；每步套环境 legal_mask 屏蔽非法动作。"""
    env.reset()
    h = None
    prev = torch.full((1, 1), policy.bos, dtype=torch.long, device=device)
    actions: list[int] = []
    masks: list[np.ndarray] = []
    done, reward, info = False, 0.0, {}  # type: ignore
    while not done:
        logits, h = policy(prev, h)  # (1,1,V)
        m = env.legal_mask()
        mt = torch.from_numpy(m).to(device).unsqueeze(0)  # (1,V)
        logits = logits[:, -1].masked_fill(~mt, float("-inf"))  # (1,V)
        a = torch.distributions.Categorical(logits=logits).sample()
        ai = int(a)
        actions.append(ai)
        masks.append(m)
        _, reward, done, info = env.step(ai)
        prev = a.view(1, 1)
    return {"actions": actions, "masks": np.array(masks), "reward": reward, "info": info}


def sequence_logprob_entropy(
    policy: FactorPolicy, actions: list[int], masks: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """teacher-forcing 重算这条序列被选 token 的 logprob 之和 与 总熵（带梯度）。"""
    t_len = len(actions)
    # 输入 [BOS, a0, ..., a_{T-2}] → 预测 [a0, ..., a_{T-1}]
    inp = torch.tensor([policy.bos] + actions[:-1], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = policy(inp)  # (1,T,V)
    logits = logits[0]  # (T,V)
    mt = torch.from_numpy(masks).to(device)  # (T,V) bool
    logits = logits.masked_fill(~mt, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)  # (T,V)
    act = torch.tensor(actions, device=device)
    chosen = logp[torch.arange(t_len), act]  # (T,) 被选动作的 logprob
    ent = -(logp.exp() * logp).nan_to_num(0.0).sum(-1)  # (T,) 每步熵（鼓励探索）
    return chosen.sum(), ent.sum()


if __name__ == "__main__":
    # 自测：假数据 + 策略，验证「能生成合法因子 + 能算 logprob + 梯度可回传」
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=80)
    stocks = [f"s{i}" for i in range(30)]
    terminals = ["open", "high", "low", "close", "volume", "amount"]
    wide = {
        t: pd.DataFrame(rng.lognormal(size=(80, 30)), index=dates, columns=stocks)
        for t in terminals
    }
    fwd = pd.DataFrame(rng.normal(0, 0.02, size=(80, 30)), index=dates, columns=stocks)
    env = FactorEnv(wide, fwd, dates[0], dates[-1] + pd.Timedelta(days=1), terminals=terminals)

    device = "cpu"
    policy = FactorPolicy(env.vocab_size).to(device)
    ep = rollout(policy, env, device)
    print("生成:", ep["info"].get("expr"), "| 奖励|IC| =", round(ep["reward"], 4))

    lp, ent = sequence_logprob_entropy(policy, ep["actions"], ep["masks"], device)
    print("logprob =", round(float(lp), 3), "| entropy =", round(float(ent), 3))
    lp.backward()  # 梯度能回传 = 训练管线通
    print("✅ 策略自测通过：生成合法因子 + 算 logprob + 梯度可回传")
