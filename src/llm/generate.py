"""LLM 因子生成·生成器：单次调用 deepseek-v4-pro（OpenAI 兼容接口）提一批因子表达式，解析成 Node。

不是 agent——就是「一次 chat.completions + 解析」的最简单 tier。
提示词在 prompts/factor_generation.txt（与代码解耦）；key/base_url 走 .env。
落盘成 factor_library schema 交给 runner，下游 screen/deflated 复用。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.core.tree import Node
from src.llm.clean import canon_key
from src.llm.parser import CORE_TERMINALS, ParseError, parse
from src.llm.prompts import load_prompt
from src.llm.usage import Usage

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # 同 loader.py：导入即加载 .env
_log = logging.getLogger(__name__)

MODEL = "deepseek-v4-pro"


def build_prompt(n: int, terminals: tuple[str, ...] = CORE_TERMINALS) -> str:
    return load_prompt("factor_generation", n=n, terminals=", ".join(terminals))


def _default_client() -> OpenAI:
    """从 .env 建 OpenAI 兼容客户端（指向 deepseek-v4-pro 接入点）。"""
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )


def call_llm(
    prompt: str,
    *,
    model: str = MODEL,
    client: OpenAI | None = None,
    temperature: float = 1.0,
    max_tokens: int = 16000,
    usage: Usage | None = None,
) -> str:
    """发一条 prompt，返回原始文本"""
    client = client or _default_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if usage is not None:
        usage.add(resp.usage)
    return resp.choices[0].message.content or ""


def parse_response(
    text: str,
    terminals: tuple[str, ...] = CORE_TERMINALS,
    *,
    logger: logging.Logger | None = None,
) -> list[Node]:
    """LLM 原始文本逐行解析成 Node，非法行跳过。不去重——去重交给 clean。"""
    log = logger or _log
    trees: list[Node] = []
    n_raw = 0
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line)  # 去编号/项目符号
        if not line:
            continue
        n_raw += 1
        try:
            trees.append(parse(line, terminals))
        except ParseError as e:
            log.debug("解析失败跳过: %s (%s)", line, e)
    log.info("LLM 提取: raw=%d parsed=%d", n_raw, len(trees))
    return trees


def generate_trees(
    n: int = 50,
    *,
    model: str = MODEL,
    terminals: tuple[str, ...] = CORE_TERMINALS,
    client: OpenAI | None = None,
    temperature: float = 1.0,
    logger: logging.Logger | None = None,
) -> list[Node]:
    """单次调用（冷启 prompt）提 n 个因子，解析+去重——standalone / 单测用。"""
    text = call_llm(build_prompt(n, terminals), model=model, client=client, temperature=temperature)
    seen: set[str] = set()
    out: list[Node] = []
    for t in parse_response(text, terminals, logger=logger):
        if (k := canon_key(t)) not in seen:
            seen.add(k)
            out.append(t)
    return out


if __name__ == "__main__":
    # 模块 logger 默认没 handler，单跑得 basicConfig 一下才看得到 INFO 计数
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build_prompt(5))
    print("\n--- 配好 .env 的 DEEPSEEK_* 后，下面会实跑（调 API、扣额度）---")
    for t in generate_trees(10):
        print(t)
