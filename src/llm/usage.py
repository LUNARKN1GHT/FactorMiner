"""LLM token / 金额累加器：一次实验持有一个，所有调用往里加，结束落盘。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, resp_usage) -> None:
        """喂 resp.usage（OpenAI 兼容返回里的 usage 对象），None 安全跳过。"""
        if resp_usage is None:
            return
        self.prompt_tokens += getattr(resp_usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(resp_usage, "completion_tokens", 0) or 0
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, price_in: float, price_out: float) -> float:
        """price_*：每百万 token 单价。返回估算金额。"""
        return self.prompt_tokens / 1e6 * price_in + self.completion_tokens / 1e6 * price_out

    def as_dict(self, price_in: float = 0.0, price_out: float = 0.0) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "est_cost": round(self.cost(price_in, price_out), 4),
        }
