"""提示词加载器：从仓库根目录 prompts/ 读 .txt 模板并填充占位符。

占位符用 $name 形式（string.Template），prompt 里可自由写 {} 不冲突。
一个 prompt 一个文件，按用途命名（如 factor_generation.txt）。
"""

from __future__ import annotations

from pathlib import Path
from string import Template

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str, **kwargs: object) -> str:
    """读 prompts/{name}.txt，用 kwargs 安全填充 $占位符。"""
    text = (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).safe_substitute(**kwargs) if kwargs else text
