"""清洗：把 LLM A 吐出的 Node 列表，砍掉太复杂的、跨轮重复的，得到值得回测的新因子。

退化因子（x−x、同窗口自减等）不在这里管——它们求值会得到 NaN/常数 IC，
在回测阶段天然被丢（见 run_llm）。这里只做便宜的句法过滤，把贵的回测留给值得算的因子。
"""

from __future__ import annotations

from src.core.tree import Node


def canon_key(node: Node) -> str:
    """规范化键：带上时序窗口。跨轮去重，避雷清单都用它

    注意不能使用 str(Node) —— 不显示时序窗口，会把不同窗口的误判成重复
    """
    if not node.children:
        return node.name  # type: ignore
    inner = ",".join(canon_key(c) for c in node.children)
    w = "" if node.value is None else f"@{node.value}"
    return f"{node.name}{w}({inner})"


def n_operators(node: Node) -> int:
    """算子非终端节点数，用来限制复杂度"""
    if not node.children:
        return 0
    return 1 + sum(n_operators(c) for c in node.children)


def clean(
    trees: list[Node],
    *,
    seen: set[str],
    max_operators: int,
) -> list[Node]:
    """过滤过度复杂、跨轮重复的因子，就地更新 seen，返回值得回测的新因子。"""
    kept: list[Node] = []
    for t in trees:
        if n_operators(t) > max_operators:  # 复杂度上限
            continue
        k = canon_key(t)
        if k in seen:  # 跨轮重复，跳过
            continue
        seen.add(k)
        kept.append(t)
    return kept
