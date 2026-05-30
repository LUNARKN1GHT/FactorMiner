"""因子表达式树结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """表达式树节点。"""

    name: str
    """节点名称"""

    arity: int = 0
    """度的数量，相当于子节点数量"""

    children: list[Node] = field(default_factory=list)
    """子节点列表"""

    value: Any = None
    """叶节点的常量值或特征名"""

    @property
    def is_leaf(self) -> bool:
        """判断该节点是否为叶子节点"""
        return self.arity == 0

    def __str__(self) -> str:
        """打印当前树的所有节点信息"""
        if self.is_leaf:
            return str(self.value if self.value is not None else self.name)
        args = ", ".join(str(c) for c in self.children)
        return f"{self.name}({args})"

    def depth(self) -> int:
        """获取当前树的深度"""
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """获取当前树的节点数量"""
        if self.is_leaf:
            return 1
        return 1 + sum(c.size() for c in self.children)


def collect_nodes(node: Node) -> list[Node]:
    """扁平化收集所有节点。"""
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_nodes(child))
    return nodes
