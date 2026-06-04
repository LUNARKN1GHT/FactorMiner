"""NSGA-II 多目标核心：非支配排序 + 拥挤度。

约定：传入的 objectives 每一维都是「越大越好」。
我们的两个目标 = (|ICIR|, -size)，把 min size 转成 max(-size)。
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable
from pathlib import Path

from src.gp.engine import GPConfig, _within_limit, crossover, mutate, ramped_half_and_half
from src.gp.tree import Node
from src.utils.logger import log_generation

# 一个个体的多目标向量
Objective = tuple[float, ...]


def _eval_objs(pop: list[Node], objective_fn: Callable[[Node], float | None]) -> list[Objective]:
    """对种群逐个算目标向量 (|ICIR|, -size)；算崩的个体给极差 |ICIR|=-1。

    Args:
        pop: 个体（表达式树）列表。
        objective_fn: 接收一棵树、返回 |ICIR| 或 None（无效）的函数。

    Returns:
        与 pop 等长的目标向量列表。
    """
    out: list[Objective] = []
    for t in pop:
        icir = objective_fn(t)
        out.append((icir if icir is not None else -1.0, -float(t.size())))
    return out


def _rank_and_crowd(objs: list[Objective]) -> tuple[dict[int, int], dict[int, float]]:
    """给每个个体标注 (层号 rank, 拥挤度 crowd)，供锦标赛比较。

    Args:
        objs: 全种群目标向量列表。

    Returns:
        (rank, crowd) 两个字典，键为个体下标。
    """
    rank, crowd = {}, {}
    for r, front in enumerate(non_dominated_sort(objs)):
        cd = crowding_distance(front, objs)
        for i in front:
            rank[i], crowd[i] = r, cd[i]
    return rank, crowd


def _tournament(rank: dict[int, int], crowd: dict[int, float], k: int) -> int:
    """拥挤比较锦标赛：抽 k 个，层低者胜；同层则拥挤度大者胜。

    Args:
        rank: 个体 → 层号。
        crowd: 个体 → 拥挤度。
        k: 锦标赛规模。

    Returns:
        获胜个体的下标。
    """
    best = random.randrange(len(rank))
    for _ in range(k - 1):
        i = random.randrange(len(rank))
        if rank[i] < rank[best] or (rank[i] == rank[best] and crowd[i] > crowd[best]):
            best = i
    return best


def dominates(a: Objective, b: Objective) -> bool:
    """判断 a 是否「支配」b。

    支配 = a 在所有目标上都不差于 b，且至少一个目标上严格更好。

    Args:
        a: 个体 a 的目标向量。
        b: 个体 b 的目标向量。

    Returns:
        a 支配 b 时返回 True。
    """
    return all(x >= y for x, y in zip(a, b, strict=True)) and any(
        x > y for x, y in zip(a, b, strict=True)
    )


def non_dominated_sort(objs: list[Objective]) -> list[list[int]]:
    """快速非支配排序：把种群按支配关系分层。

    Args:
        objs: 每个个体的目标向量列表，下标即个体编号。

    Returns:
        分层结果 [front0, front1, ...]，每层是个体下标列表；
        front0 即 Pareto 前沿（不被任何个体支配）。
    """
    n = len(objs)
    dominated: list[list[int]] = [[] for _ in range(n)]  # i 支配了哪些
    dom_count = [0] * n  # 有多少个支配 i
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if dominates(objs[p], objs[q]):
                dominated[p].append(q)
            elif dominates(objs[q], objs[p]):
                dom_count[p] += 1

        if dom_count[p] == 0:
            # 不被任何支配
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()  # 最后一层是空的，丢掉
    return fronts


def crowding_distance(front: list[int], objs: list[Objective]) -> dict[int, float]:
    """计算同一层内每个个体的拥挤度（越大越稀疏、越该保留）。

    Args:
        front: 该层的个体下标列表。
        objs: 全种群的目标向量列表。

    Returns:
        {个体下标: 拥挤度}；边界个体为 inf（必保留以撑开前沿）。
    """
    dist: dict[int, float] = {i: 0.0 for i in front}
    if len(front) <= 2:
        # 太少，全给无穷大
        return {i: float("inf") for i in front}

    n_obj = len(objs[0])
    for m in range(n_obj):
        # 每个目标维度分别累加
        order = sorted(front, key=lambda i: objs[i][m])
        dist[order[0]] = dist[order[-1]] = float("inf")
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        if hi == lo:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += (objs[order[k + 1]][m] - objs[order[k - 1]][m]) / (hi - lo)

    return dist


def run_gp_nsga2(
    objective_fn: Callable[[Node], float | None],
    config: GPConfig | None = None,
    logger: logging.Logger | None = None,
    log_dir: Path | None = None,
) -> list[tuple[Node, float, int]]:
    """多目标 GP 进化主循环：同时优化 (max |ICIR|, min size)。

    Args:
        objective_fn: 接收一棵树、返回 |ICIR| 或 None（无效因子）的函数。
        config: GP 超参数配置。
        logger: 实验日志，None 时退回 print。
        log_dir: 实验目录，传了才写 stats.jsonl。

    Returns:
        最终 Pareto 前沿，每项为 (表达式树, |ICIR|, size)，按 size 升序。
    """
    config = config or GPConfig()

    # 强类型开关：选类型化 / 普通三件套
    if config.strongly_typed:
        from src.gp.stgp import typed_crossover, typed_mutate, typed_ramped_half_and_half

        init_fn, crossover_fn, mutate_fn = (
            typed_ramped_half_and_half,
            typed_crossover,
            typed_mutate,
        )
    else:
        init_fn, crossover_fn, mutate_fn = ramped_half_and_half, crossover, mutate

    pop = init_fn(config.population_size, config.min_depth, config.max_depth)
    objs = _eval_objs(pop, objective_fn)

    for gen in range(config.generations):
        rank, crowd = _rank_and_crowd(objs)

        # 1. 生成子代 Q
        offspring: list[Node] = []
        while len(offspring) < config.population_size:
            i = _tournament(rank, crowd, config.tournament_size)
            if random.random() < config.crossover_prob:
                j = _tournament(rank, crowd, config.tournament_size)
                for c in crossover_fn(pop[i], pop[j]):
                    offspring.append(
                        c if _within_limit(c, config.max_depth) else copy.deepcopy(pop[i])
                    )
            else:
                c = copy.deepcopy(pop[i])
                if random.random() < config.mutation_prob:
                    m = mutate_fn(c, max_depth=3)
                    c = m if _within_limit(m, config.max_depth) else c
                offspring.append(c)
        offspring = offspring[: config.population_size]

        # 2. 合并 P 和 Q 的到 R
        r = pop + offspring
        r_objs = objs + _eval_objs(offspring, objective_fn)

        # 基因型去重：同一个表达式只保留一份
        uniq: dict = {}
        for (
            t,
            o,
        ) in zip(r, r_objs, strict=False):
            uniq.setdefault(str(t), (t, o))
        r = [t for t, _ in uniq.values()]
        r_objs = [o for _, o in uniq.values()]

        # 3. 环境选择：按层装，最后一层用拥挤度补满
        chosen: list[int] = []
        for front in non_dominated_sort(r_objs):
            if len(chosen) + len(front) <= config.population_size:
                chosen.extend(front)
            else:
                cd = crowding_distance(front, r_objs)
                front.sort(key=lambda i: cd[i], reverse=True)  # 稀疏优先
                chosen.extend(front[: config.population_size - len(chosen)])
                break

        pop = [r[i] for i in chosen]
        objs = [r_objs[i] for i in chosen]

        # -- 日志 --
        f0 = non_dominated_sort(objs)[0]
        best_i = max(f0, key=lambda i: objs[i][0])
        best_icir = objs[best_i][0]
        mean_icir = sum(o[0] for o in objs) / len(objs)
        sizes = sorted(int(-objs[i][1]) for i in f0)

        msg = (
            f"Gen {gen:3d} | front0={len(f0):2d} | best|ICIR|={best_icir:.4f} "
            f"| mean|ICIR|={mean_icir:.4f} | sizes={sizes}"
        )
        logger.info(msg) if logger is not None else print(msg)

        if log_dir is not None:
            log_generation(
                log_dir,
                gen=gen,
                best_ic=best_icir,
                mean_ic=mean_icir,
                best_expr=str(pop[best_i]),
                extra={
                    "metric": "abs_icir",
                    "method": "nsga2",
                    "front0_size": len(f0),
                    "pareto_sizes": sizes,
                },
            )

    # 返回最终 Pareto 前沿，同时保证打印不重复
    seen: set = set()
    pareto: list = []
    for i in non_dominated_sort(objs)[0]:
        key = str(pop[i])
        if key in seen:
            continue
        seen.add(key)
        pareto.append((pop[i], objs[i][0], int(-objs[i][1])))
    pareto.sort(key=lambda x: x[2])
    return pareto
