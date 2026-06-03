"""一个字段（如 close）单独抽出来，就得到一个宽表。六个字段就是六个宽表。便于计算"""

import pandas as pd

from src.gp.operators import BINARY_OPS, TS_OPS, UNARY_OPS
from src.gp.tree import Node


def to_wide(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """把 loader 的长表 (MultiIndex date, code) 拆成 「字段 -> 宽表」字典。

    每张宽表: index=date, columns=code.
    这样后面所有算子都在统一形态的宽表上运算。
    """
    # prices.columns 形如 [open, high, low, close, volume, amount, turnover]
    # prices[field] 是一个 (date,code) 双索引的 Series，
    # .unstack("code") 把 code 这层索引「摊开」成列 → 得到 date × code 宽表
    return {field: prices[field].unstack("code") for field in prices.columns}


# 把三张算子表合并成「算子名 -> 函数」的查找表
_FUNCS = {name: fn for name, (fn, _) in {**BINARY_OPS, **UNARY_OPS, **TS_OPS}.items()}
_TS_NAMES = set(TS_OPS)


def evaluate(node: Node, wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把一颗表达式树求值成因子宽表 (date x code).

    wide: to_wide() 产出的「字段 -> 宽表」字典
    返回：一张 date x code 的因子值宽表
    """
    # 1. 叶子节点：直接返回对应字段的宽表
    if node.is_leaf:
        return wide[node.value]

    fn = _FUNCS[node.name]

    # 2. 时序算子：只有一个子树，窗口长度在 node.value 里
    if node.name in _TS_NAMES:
        x = evaluate(node=node.children[0], wide=wide)  # 先递归算出子序列
        # 带窗口调用 ts_
        return fn(x, int(node.value))  # type:ignore

    # 算术 / 一元 / 截面算子：先算出所有子树，再当前位置参数传入
    args = [evaluate(c, wide=wide) for c in node.children]
    # 传入算子
    return fn(*args)  #  type: ignore
