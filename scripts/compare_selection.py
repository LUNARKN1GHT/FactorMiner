"""离线对比不同「选因子准则」的样本外表现——只读 walkforward.pkl，不重跑 GP。

walk-forward 每折存下了整条 Pareto 前沿（每个个体的 train/test ICIR + 表达式树）。
原脚本默认按「训练段 |ICIR| 最强」挑每折代表，但结果显示这会系统性挑到过拟合的复杂因子。
这里把同一批前沿，换几种选法重新挑一遍，看谁的 OOS 更高、更稳。
"""

import pickle
import statistics
import sys
from collections.abc import Callable

# 一种「选法」= 给定本折所有个体(rows)，返回选中的那一个 row
Rule = Callable[[list[dict]], dict]


def by_train_max(rows: list[dict]) -> dict:
    """基线：训练段 |ICIR| 最强——walk-forward 原本就这么选。"""
    return max(rows, key=lambda r: abs(r["train_icir"]))


def by_min_size(rows: list[dict]) -> dict:
    """纯简约：选最小的树；size 相同再比训练段 |ICIR|。

    元组排序的小技巧：先按 size 升序，size 打平时用 -|ICIR| 让强的排前面，
    min() 取到的就是「最小且其中最强」的那个。
    """
    return min(rows, key=lambda r: (r["size"], -abs(r["train_icir"])))


def by_parsimony(rows: list[dict], penalty: float = 0.02) -> dict:
    """简约惩罚：在 |ICIR| 里扣掉 penalty*size，越复杂扣得越狠。

    penalty 是每多一个节点要「值多少 ICIR」才划算——0.02 是拍脑袋的起点，
    可以调大（更偏爱简单）或调小（更接近 train-max）来看选择怎么变。
    """
    return max(rows, key=lambda r: abs(r["train_icir"]) - penalty * r["size"])


def summarize(name: str, rule: Rule, folds: list[dict]) -> None:
    """对一种选法：逐折打印选中因子的战绩，并报 OOS |ICIR| 的均值/标准差/最差。"""
    print(f"\n=== 选法：{name} ===")
    print(f"{'fold':>4} {'size':>4} {'train ICIR':>12} {'test ICIR':>12} {'衰减':>6}  expr")
    oos = []  # 各折选中因子的 test|ICIR|（绝对值），用来算汇总离散度
    for fold in folds:
        pick = rule(fold["rows"])
        # 带符号打印方向，符号相反则标翻转——和 walk-forward 汇总同一套口径
        flip = " ⚠符号翻转" if pick["train_icir"] * pick["test_icir"] < 0 else ""
        print(
            f"{fold['fold']:>4} {pick['size']:>4} "
            f"{pick['train_icir']:>+12.4f} {pick['test_icir']:>+12.4f} "
            f"{pick['decay']:>5.0f}%  {pick['expr']}{flip}"
        )
        oos.append(abs(pick["test_icir"]))
    print(
        f"OOS |ICIR| 各折={[round(x, 4) for x in oos]} | "
        f"均值={statistics.mean(oos):.4f} 标准差={statistics.pstdev(oos):.4f} 最差={min(oos):.4f}"
    )


def main(pkl_path: str) -> None:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    folds = payload["folds"]

    # 想加选法只需在这里挂一行，summarize 一视同仁
    rules: dict[str, Rule] = {
        "train-max（基线）": by_train_max,
        "min-size（纯简约）": by_min_size,
        "parsimony（简约惩罚）": by_parsimony,
    }
    for name, rule in rules.items():
        summarize(name, rule, folds)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python scripts/compare_selection.py <walkforward.pkl 路径>")
    main(sys.argv[1])
