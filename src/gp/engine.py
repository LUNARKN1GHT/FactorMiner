"""GP 进化引擎：选择、交叉、变异。"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from src.gp.operators import BINARY_OPS, TS_OPS, UNARY_OPS
from src.gp.tree import Node, collect_nodes
from src.utils.logger import log_generation


@dataclass
class GPConfig:
    population_size: int = 500
    generations: int = 50
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    max_depth: int = 6
    min_depth: int = 2


# ---- 终端集（叶节点） ----

TERMINALS = ["open", "high", "low", "close", "volume", "amount"]
TS_WINDOWS = [5, 10, 20]


def random_terminal() -> Node:
    name = random.choice(TERMINALS)
    return Node(name=name, arity=0, value=name)


# ---- 随机树生成 ----

TS_OP_NAMES = set(TS_OPS)


def random_tree(max_depth: int, min_depth: int = 0) -> Node:
    """Grow 方法生成随机表达式树。"""

    # 迭代出口，处理掉异常情况
    if max_depth <= 0 or (max_depth <= min_depth and random.random() < 0.5):
        return random_terminal()

    # 定义所有操作的集合
    all_ops = list(BINARY_OPS.items()) + list(UNARY_OPS.items()) + list(TS_OPS.items())
    op_name, (_, arity) = random.choice(all_ops)

    if op_name in TS_OP_NAMES:
        # 时序算子：树上只有一个字节点，窗口长度存进 value
        child = random_tree(max_depth - 1, min_depth - 1)
        return Node(name=op_name, arity=1, children=[child], value=random.choice(TS_WINDOWS))

    # 算术（二元）/一元/截面算子：通常按arity生成子树
    children = [random_tree(max_depth - 1, min_depth - 1) for _ in range(arity)]
    return Node(name=op_name, arity=arity, children=children)


# ---- 遗传操作 ----


def tournament_select(population: list[Node], fitnesses: np.ndarray, k: int) -> Node:
    indices = random.sample(range(len(population)), k)
    best = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best])


def crossover(parent1: Node, parent2: Node) -> tuple[Node, Node]:
    """子树交叉。"""
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)

    nodes1 = collect_nodes(p1)
    nodes2 = collect_nodes(p2)

    point1 = random.choice(nodes1)
    point2 = random.choice(nodes2)

    point1.name, point2.name = point2.name, point1.name
    point1.arity, point2.arity = point2.arity, point1.arity
    point1.children, point2.children = point2.children, point1.children
    point1.value, point2.value = point2.value, point1.value

    return p1, p2


def mutate(individual: Node, max_depth: int = 3) -> Node:
    """子树变异：随机替换一个子树。"""
    ind = copy.deepcopy(individual)
    nodes = collect_nodes(ind)
    point = random.choice(nodes)
    new_subtree = random_tree(max_depth)

    point.name = new_subtree.name
    point.arity = new_subtree.arity
    point.children = new_subtree.children
    point.value = new_subtree.value

    return ind


# ---- 主循环 ----


def run_gp(
    fitness_fn,
    config: GPConfig | None = None,
    logger=None,
    log_dir=None,
) -> list[tuple[Node, float]]:
    """GP 进化主循环。

    Args:
        fitness_fn: 适应度函数，接收 Node 返回 float（越大越好）。
        config: GP 超参数配置。
        logger: 实验日志，None 时退回 print
        log_dif: 实验目录

    Returns:
        按适应度降序排列的 (个体, 适应度) 列表。
    """
    if config is None:
        config = GPConfig()

    population = [
        random_tree(config.max_depth, config.min_depth) for _ in range(config.population_size)
    ]

    best_history = []

    for gen in range(config.generations):
        fitnesses = np.array([fitness_fn(ind) for ind in population])

        best_idx = np.argmax(fitnesses)
        best_history.append((copy.deepcopy(population[best_idx]), fitnesses[best_idx]))

        msg = (
            f"Gen {gen:3d} | best={fitnesses[best_idx]:.4f} "
            f"| mean={fitnesses.mean():.4f} | expr={population[best_idx]}"
        )
        logger.info(msg) if logger is not None else print(msg)

        # 结构化记录这一代，供事后分析（best_ic/mean_ic 字段这里装的是 |ICIR| 适应度）
        if log_dir is not None:
            log_generation(
                log_dir,
                gen=gen,
                best_ic=float(fitnesses[best_idx]),
                mean_ic=float(fitnesses.mean()),
                best_expr=str(population[best_idx]),
                extra={"metric": "abs_icir", "pop_size": config.population_size},
            )

        next_pop = [copy.deepcopy(population[best_idx])]  # 精英保留

        while len(next_pop) < config.population_size:
            if random.random() < config.crossover_prob:
                p1 = tournament_select(population, fitnesses, config.tournament_size)
                p2 = tournament_select(population, fitnesses, config.tournament_size)
                c1, c2 = crossover(p1, p2)
                next_pop.extend([c1, c2])
            else:
                p = tournament_select(population, fitnesses, config.tournament_size)
                if random.random() < config.mutation_prob:
                    p = mutate(p, max_depth=3)
                next_pop.append(p)

        population = next_pop[: config.population_size]

    results = sorted(best_history, key=lambda x: x[1], reverse=True)
    return results
