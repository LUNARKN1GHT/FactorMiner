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
from src.llm.parser import CORE_TERMINALS, ParseError, parse
from src.llm.prompts import load_prompt

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


def _canon_key(node: Node) -> str:
    """递归规范化键：带上时序窗口，避免不同窗口被误判重复。"""
    if not node.children:
        return node.name  # type: ignore
    inner = ",".join(_canon_key(c) for c in node.children)
    w = "" if node.value is None else f"@{node.value}"
    return f"{node.name}{w}({inner})"


def generate_trees(
    n: int = 50,
    *,
    model: str = MODEL,
    terminals: tuple[str, ...] = CORE_TERMINALS,
    client: OpenAI | None = None,
    temperature: float = 1.0,  # 调高些促多样性,
    logger: logging.Logger | None = None,
) -> list[Node]:
    """单次调用 LLM 提 n 个因子，解析成 Node（非法/重复跳过）。"""
    log = logger or _log
    client = client or _default_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(n, terminals)}],
        max_tokens=16000,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""

    trees: list[Node] = []
    seen: set[str] = set()
    n_raw = n_parsed = 0
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line)  # 去掉可能的编号/项目符号
        if not line:
            continue
        n_raw += 1
        try:
            tree = parse(line, terminals)
        except ParseError as e:
            log.debug("解析失败跳过: %s (%s)", line, e)  # 调 prompt 时开 DEBUG 看 LLM 犯啥错
            continue
        n_parsed += 1
        if (k := _canon_key(tree)) not in seen:
            seen.add(k)
            trees.append(tree)
    log.info("LLM 提取: raw=%d parsed=%d unique=%d", n_raw, n_parsed, len(trees))
    return trees


if __name__ == "__main__":
    # 模块 logger 默认没 handler，单跑得 basicConfig 一下才看得到 INFO 计数
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build_prompt(5))
    print("\n--- 配好 .env 的 DEEPSEEK_* 后，下面会实跑（调 API、扣额度）---")
    for t in generate_trees(10):
        print(t)
