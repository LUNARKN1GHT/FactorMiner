"""因子表达式树结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """表达式树节点。"""

    name: str
    arity: int = 0
    children: list[Node] = field(default_factory=list)
    value: Any = None  # 叶节点的常量值或特征名

    @property
    def is_leaf(self) -> bool:
        return self.arity == 0

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.value if self.value is not None else self.name)
        args = ", ".join(str(c) for c in self.children)
        return f"{self.name}({args})"

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(c.size() for c in self.children)


def collect_nodes(node: Node) -> list[Node]:
    """扁平化收集所有节点。"""
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_nodes(child))
    return nodes
