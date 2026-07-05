"""AlphaGen (KDD 2023, RL-MLDM/alphagen) 复现·表达式桥接。

官方仓库跑在 qlib/baostock 的 CSI300 数据上，跟本项目的 tushare/hs300 不是同一套
数据管线——"官方口径"的数字和这里桥接后重新算出来的数字天然不完全等价，因此分两份
报告，不混在一起充当一个数字（见 `doc/AlphaGen复现.md`）。这个模块只做一件事：把
官方仓库挖出的表达式字符串（`Expression.__str__()` 的输出，如
``Mean($close,5d)``）解析成本项目的 `Node`，方便复用自己的
`src/core/evaluator.py` 在自己的数据上重新评估、进而喂 `compare_generators.py`。

算子映射：能一一对上的照搬。AlphaGen 有而本项目 `operators.py` 没有的算子
（Sign/Pow/Greater/Less/Sum/Var/Skew/Kurt/Med/Mad/WMA/EMA/Cov），以及用到 ``$vwap``
终端（本项目宽表没有 vwap 字段）或数值常量的表达式，一律判不可翻译并计数，不为了
凑覆盖率强行扩算子——翻译不了是预期内的信息损失，如实报告覆盖率就好。
"""

from __future__ import annotations

import re

from src.core.tree import Node

_UNARY = {"Abs": "abs", "Log": "log", "CSRank": "rank"}
"""一元算子：AlphaGen 名 -> 本项目 operators.py 里的算子名"""

_BINARY = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div"}
"""二元算子（非时序）：AlphaGen 名 -> 本项目算子名"""

_TS_UNARY = {
    "Ref": "delay",
    "Mean": "ts_mean",
    "Std": "ts_std",
    "Delta": "delta",
    "Max": "ts_max",
    "Min": "ts_min",
    "Rank": "ts_rank",
}
"""时序一元算子（数据 + 窗口）：AlphaGen 的 RollingOperator -> 本项目 TS_OPS"""

_TS_BINARY = {"Corr": "ts_corr"}
"""时序二元算子（两个数据 + 窗口）：AlphaGen 的 PairRollingOperator -> 本项目 TS_BINARY_OPS"""

_TERMINALS = {"open", "close", "high", "low", "volume"}
"""AlphaGen 终端里本项目宽表也有的字段；``vwap`` 本项目没有，遇到判不可翻译"""

_UNSUPPORTED = {
    "Sign", "Pow", "Greater", "Less", "Sum", "Var", "Skew", "Kurt", "Med", "Mad", "WMA", "EMA", "Cov",
}
"""AlphaGen 有、本项目 operators.py 没有对应实现的算子"""


class UnsupportedExprError(ValueError):
    """表达式用到了算子映射表覆盖不到的算子 / 终端 / 数值常量。"""


def _split_top_level_args(inner: str) -> list[str]:
    """按顶层逗号切分函数参数，跳过嵌套括号内的逗号。"""
    args, depth, start = [], 0, 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(inner[start:i])
            start = i + 1
    args.append(inner[start:])
    return args


def _parse_delta(arg: str) -> int:
    """解析 AlphaGen 的时间增量记法（如 ``5d``）为整数窗口长度。"""
    arg = arg.strip()
    if not arg.endswith("d"):
        raise UnsupportedExprError(f"不支持的时间增量记法：{arg}")
    return int(arg[:-1])


def parse_alphagen_expr(expr: str) -> Node:
    """把 AlphaGen 的表达式字符串解析成本项目的 `Node`。

    Args:
        expr (str): AlphaGen `Expression.__str__()` 的输出，形如
            ``Op(arg1,arg2)``、终端 ``$field``、时间增量 ``Nd``。

    Raises:
        UnsupportedExprError: 用到了不在算子映射表内的算子 / 终端 / 数值常量。

    Returns:
        Node: 翻译后的表达式树，可直接喂 `src.core.evaluator.evaluate`。
    """
    expr = expr.strip()
    if expr.startswith("$"):
        field = expr[1:]
        if field not in _TERMINALS:
            raise UnsupportedExprError(f"不支持的终端：${field}（本项目宽表无此字段）")
        return Node(name=field, arity=0, value=field)

    m = re.fullmatch(r"([A-Za-z]+)\((.*)\)", expr)
    if not m:
        # 剩下的情形只有数值常量（如 "5.0"）——本项目算子表没有常量叶子节点
        raise UnsupportedExprError(f"不支持的数值常量：{expr}")

    op, inner = m.group(1), m.group(2)
    args = _split_top_level_args(inner)

    if op in _UNARY and len(args) == 1:
        child = parse_alphagen_expr(args[0])
        return Node(name=_UNARY[op], arity=1, children=[child])

    if op in _BINARY and len(args) == 2:
        lhs, rhs = (parse_alphagen_expr(a) for a in args)
        return Node(name=_BINARY[op], arity=2, children=[lhs, rhs])

    if op in _TS_UNARY and len(args) == 2:
        child = parse_alphagen_expr(args[0])
        window = _parse_delta(args[1])
        return Node(name=_TS_UNARY[op], arity=1, children=[child], value=window)

    if op in _TS_BINARY and len(args) == 3:
        lhs, rhs = parse_alphagen_expr(args[0]), parse_alphagen_expr(args[1])
        window = _parse_delta(args[2])
        return Node(name=_TS_BINARY[op], arity=2, children=[lhs, rhs], value=window)

    if op in _UNSUPPORTED:
        raise UnsupportedExprError(f"不支持的算子：{op}（本项目 operators.py 无对应实现）")
    raise UnsupportedExprError(f"未知算子：{op}")


def translate_pool(exprs: list[str]) -> tuple[list[tuple[str, Node]], list[tuple[str, str]]]:
    """批量翻译一个因子池的表达式列表，成功/失败分别收集。

    Args:
        exprs (list[str]): AlphaGen 因子池导出的表达式字符串列表（如
            `to_json_dict()["exprs"]`）。

    Returns:
        tuple[list[tuple[str, Node]], list[tuple[str, str]]]:
            - 翻译成功的 ``(原字符串, Node)`` 列表。
            - 翻译失败的 ``(原字符串, 失败原因)`` 列表，用于如实报告覆盖率。
    """
    ok: list[tuple[str, Node]] = []
    failed: list[tuple[str, str]] = []
    for expr in exprs:
        try:
            ok.append((expr, parse_alphagen_expr(expr)))
        except UnsupportedExprError as e:
            failed.append((expr, str(e)))
    return ok, failed


if __name__ == "__main__":
    # 自测：覆盖每类算子（一元/二元/时序一元/时序二元）+ 不可翻译情形（vwap/未知算子/常量）
    cases_ok = {
        "$close": "close",
        "Abs($close)": "abs(close)",
        "Add($open,$close)": "add(open, close)",
        "Mean($close,5d)": "ts_mean(close, 5)",
        "Ref($close,10d)": "delay(close, 10)",
        "Max($high,20d)": "ts_max(high, 20)",
        "Corr($close,$volume,10d)": "ts_corr(close, volume, 10)",
        "Div(Mean($close,5d),CSRank($volume))": "div(ts_mean(close, 5), rank(volume))",
    }
    for expr, want in cases_ok.items():
        got = str(parse_alphagen_expr(expr))
        assert got == want, f"{expr} -> {got}，期望 {want}"
        print(f"OK  {expr:<35} -> {got}")

    cases_bad = ["$vwap", "Sign($close)", "5.0", "Cov($close,$volume,10d)"]
    for expr in cases_bad:
        try:
            parse_alphagen_expr(expr)
            raise AssertionError(f"{expr} 应该翻译失败但没有报错")
        except UnsupportedExprError as e:
            print(f"跳过 {expr:<25} -> {e}")

    ok, failed = translate_pool([*cases_ok, *cases_bad])
    assert len(ok) == len(cases_ok) and len(failed) == len(cases_bad)
    print(f"✅ alphagen_bridge 自测通过：{len(ok)} 条可翻译，{len(failed)} 条按预期跳过")
