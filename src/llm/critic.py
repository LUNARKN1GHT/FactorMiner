"""LLM B（评论家）：读本轮因子的训练集表现 + 当前存档，给出诊断意见。

只诊断不打分——谁活下来由 run_llm 用硬指标决定。输出是一段自由文本，
原样拼进 A 的进化 prompt。prompt 在 prompts/factor_critique.txt。
"""

from __future__ import annotations

import logging

from openai import OpenAI

from src.llm.generate import MODEL, _default_client
from src.llm.prompts import load_prompt

_log = logging.getLogger(__name__)


def _fmt(rows: list[dict], empty: str) -> str:
    """因子列表 → 纯文本，每行『表达式 | train_ic=.. size=..』。不含 test_ic（铁律）。"""
    if not rows:
        return empty
    return "\n".join(f"{r['expr']} | train_ic={r['train_ic']:.4f} size={r['size']}" for r in rows)


def critique(
    results: list[dict],
    archive: list[dict],
    *,
    model: str = MODEL,
    client: OpenAI | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """调 LLM B，返回诊断文本（三段：整体问题 / 值得深挖 / 下轮指令）。"""
    log = logger or _log
    client = client or _default_client()
    prompt = load_prompt(
        "factor_critique",
        results=_fmt(results, "（本轮无有效因子）"),
        archive=_fmt(archive, "（存档为空）"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,  # 诊断需要更冷静的内容
    )
    text = (resp.choices[0].message.content or "").strip()
    log.info("LLM B 诊断 %d 字", len(text))
    return text
