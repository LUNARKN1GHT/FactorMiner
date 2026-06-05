"""强类型 GP（STGP）：带类型的原语表 + 类型化树生成。

两类：S=带量纲序列（价量额及其变换），R=截面排名（rank 的输出，∈[0,1]）。
规则：rank 是唯一改变类型的算子（S→R，输入只能 S）；其余算子泛型 T→T（子节点同型）。
由此生成的树天然合法：禁掉 rank(rank(x))、add(close, rank(x)) 这类垃圾。
"""

from __future__ import annotations

import copy
import random

from src.gp.engine import TERMINALS, TS_BINARY_OPS, TS_WINDOWS
from src.gp.operators import BINARY_OPS, TS_OPS, UNARY_OPS
from src.gp.tree import Node, collect_nodes

TYPES = ("S", "R")

# PRODUCERS[T] = 能产出类型 T 的算子条目：(算子名, 子节点类型清单)
# 泛型算子在 S/R 两表各登记一份；rank 单独只进 R 表
PRODUCERS: dict[str, list[tuple[str, list[str]]]] = {"S": [], "R": []}
for _t in TYPES:
    for _name in BINARY_OPS:
        PRODUCERS[_t].append((_name, [_t, _t]))
    for _name in UNARY_OPS:
        if _name != "rank":
            PRODUCERS[_t].append((_name, [_t]))
    for _name in TS_OPS:
        PRODUCERS[_t].append((_name, [_t]))
    for _name in TS_BINARY_OPS:
        PRODUCERS[_t].append((_name, [_t, _t]))
PRODUCERS["R"].append(("rank", ["S"]))  # rank: S-R 是唯一的类型转换口

TS_OP_NAMES = set(TS_OPS)
TS_WINDOW_NAMES = set(TS_OPS) | set(TS_BINARY_OPS)  # 这些算子要在 value 里面存窗口


def typed_terminal() -> Node:
    """随机终端，类型恒为 S"""
    name = random.choice(TERMINALS)
    return Node(name=name, arity=0, value=name, out_type="S")


def _minimal(out_type: str) -> Node:
    """深度耗尽时该类型的最小子树"""
    if out_type == "S":
        return typed_terminal()
    return Node(name="rank", arity=1, children=[typed_terminal()], out_type="R")


def typed_random_tree(out_type: str, max_depth: int, min_depth: int = 0) -> Node:
    """Grow 生成一颗 「根输出 out_typ」 的合法树"""
    # 深度耗尽：收口到最小子树
    if max_depth <= 0:
        return _minimal(out_type=out_type)

    # 没到最小子树就强制分支：满足后，仅 S 型有一半概率收口成终端
    if min_depth <= 0 and out_type == "S" and random.random() < 0.5:
        return typed_terminal()

    name, arg_types = random.choice(PRODUCERS[out_type])
    children = [
        typed_random_tree(out_type=at, max_depth=max_depth - 1, min_depth=min_depth - 1)
        for at in arg_types
    ]
    value = random.choice(TS_WINDOWS) if name in TS_WINDOW_NAMES else None
    return Node(name=name, arity=len(children), children=children, value=value, out_type=out_type)


def recompute_type(node: Node) -> str:
    """独立地把类型自底向上重推一遍，并校验与盖章的 out_type 一致——生成器的自检工具。

    任何非法结构（rank 套非 S、子节点类型不一致、标签与推导不符）都会在这里 assert 失败。
    """
    if node.is_leaf:
        assert node.name in TERMINALS, f"未知终端 {node.name}"
        t = "S"
    elif node.name == "rank":
        ct = recompute_type(node.children[0])
        assert ct == "S", f"rank 舒徐必须 S，实得 {ct}"
        t = "R"
    else:
        cts = [recompute_type(c) for c in node.children]
        assert len(set(cts)) == 1, f"{node.name} 子节点类型不一致 {cts}"
        t = cts[0]
    assert t == node.out_type, f"{node.name} 标签 {node.out_type} 与推导 {t} 不符"
    return t


def typed_crossover(parent1: Node, parent2: Node) -> tuple[Node, Node]:
    """子树交叉，但只在「同输出类型」的子树间交换——交换后两棵树依然合法"""
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    nodes1 = collect_nodes(p1)
    nodes2 = collect_nodes(p2)

    types2 = {n.out_type for n in nodes2}
    point1 = random.choice([n for n in nodes1 if n.out_type in types2])
    point2 = random.choice([n for n in nodes2 if n.out_type == point1.out_type])

    # 两端 out_type 相等，无需交换；只换结构内容（名字/度/孩子/窗口值）
    point1.name, point2.name = point2.name, point1.name
    point1.arity, point2.arity = point2.arity, point1.arity
    point1.children, point2.children = point2.children, point1.children
    point1.value, point2.value = point2.value, point1.value

    return p1, p2


def typed_mutate(individual: Node, max_depth: int = 3) -> Node:
    """子树变异：随机挑一点，替换成「同类型」的新随机子树。"""
    ind = copy.deepcopy(individual)
    point = random.choice(collect_nodes(ind))
    subtree = typed_random_tree(point.out_type, max_depth)

    point.name = subtree.name
    point.arity = subtree.arity
    point.children = subtree.children
    point.value = subtree.value
    point.out_type = subtree.out_type  # 同型，等于没改；写上更醒目
    return ind


def typed_ramped_half_and_half(pop_size: int, min_depth: int, max_depth: int) -> list[Node]:
    """Koza ramped 初始化的类型版：每棵树随机选 S 或 R 当根类型。

    原版一半 full、一半 grow；这里只有 grow 版生成器，故用 min_depth 近似深浅差异：
    偶数个体把下限拉到该档深度（长得更满），奇数个体下限=1（形状不规则）。full 版留作之后扩展。
    """
    population: list[Node] = []
    depths = list(range(min_depth, max_depth + 1))  # 深度阶梯
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        root_type = random.choice(TYPES)  # S/R 各半，丰富多样性
        low = depth if i % 2 == 0 else 1
        population.append(typed_random_tree(root_type, depth, min_depth=low))
    return population


if __name__ == "__main__":
    # 第①步的便宜验证：各类型生成几棵，recompute_type 不抛错即合法
    random.seed(0)
    for target in TYPES:
        print(f"--- 目标类型 {target} ---")
        for _ in range(6):
            tree = typed_random_tree(target, max_depth=4, min_depth=1)
            recompute_type(tree)  # 非法会在此 assert 失败
            print(f"[{tree.out_type}] {tree}")

    # 第②步：反复交叉/变异，验证合法性是否被保持
    print("--- 交叉/变异合法性压力测试 ---")
    bad = 0
    for _ in range(2000):
        a = typed_random_tree(random.choice(TYPES), max_depth=4, min_depth=1)
        b = typed_random_tree(random.choice(TYPES), max_depth=4, min_depth=1)
        c1, c2 = typed_crossover(a, b)
        m = typed_mutate(a, max_depth=3)
        for t in (c1, c2, m):
            try:
                recompute_type(t)
            except AssertionError as exc:
                bad += 1
                print("非法：", exc)
    print(f"2000 轮交叉+变异，非法 {bad} 例")
