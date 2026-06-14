"""LLM 因子生成·解析层：把 LLM 吐的函数式表达式文本解析成 Node 树（契约适配器）。

LLM 输出形如 div(rank(amount), ts_mean(close, 10))，本层解析成 core 的 Node，
按 core.operators 注册表校验算子名/参数个数/窗口，非法抛 ParseError（runner 跳过）。
纯函数、不碰 API/数据，可独立单测。按顶层设计，只依赖 core、不碰其它生成器。
"""

import re

from src.core.operators import BINARY_OPS, TS_BINARY_OPS, TS_OPS, UNARY_OPS
from src.core.tree import Node

CORE_TERMINALS = ("open", "high", "low", "close", "volume", "amount")

# 蒜子规格：name -> (tree_arity, has_window)， 从 core 注册表派生
_OP_SEPC: dict[str, tuple[int, bool]] = {
    **{n: (2, False) for n in BINARY_OPS},  # 二元算数：函数 2 参
    **{n: (1, False) for n in UNARY_OPS},  # 一元：函数 1 参
    **{n: (1, True) for n in TS_OPS},  # 一元时序：函数 2 参 (child, window)
    **{n: (2, True) for n in TS_BINARY_OPS},  # 二元时序 ts_corr: 函数 3 参 (x, y, window)
}


class ParseError(ValueError):
    """ "LLM 表达式非法（未知算子 / 参数个数错 / 窗口缺失或非法 / 语法错）。"""


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|[(),]")


def _tokenize(s: str) -> list[str]:
    """切成 token：标识符 / 整数 / 括号逗号；无法识别的字符抛 ParseError。"""
    tokens, pos = [], 0
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(s, pos=pos)
        if not m:
            raise ParseError(f"非法字符 @{pos}: {s[pos : pos + 10]!r}")
        tokens.append(m.group(0))
        pos = m.end()
    return tokens


def _build_op(name: str, args: list) -> Node:
    """按规格校验并建算子节点（窗口存进 Node.value，对齐 GP/RL 约定）。"""
    if name not in _OP_SEPC:
        raise ParseError(f"未知算子 {name!r}")
    tree_arity, has_widow = _OP_SEPC[name]
    n_expected = tree_arity + (1 if has_widow else 0)
    if len(args) != n_expected:
        raise ParseError(f"{name} 需 {n_expected} 个参数，得到 {len(args)}")

    window, children = None, args
    if has_widow:
        window, children = args[-1], args[:-1]
        if not isinstance(window, int):
            raise ParseError(f"{name} 最后一参应是窗口整数，得到 {window!r}")
        if window < 2:
            raise ParseError(f"{name} 窗口需要 ≥ 2，得到 {window}")
    for c in children:
        if not isinstance(c, Node):
            raise ParseError(f"{name} 的参数应是子表达式，得到裸整数 {c!r}")
    return Node(name=name, arity=tree_arity, children=children, value=window)


def parse(expr: str, terminals: tuple[str, ...] = CORE_TERMINALS) -> Node:
    """把函数式表达式文本解析成 Node。非法抛 ParseError。"""
    tokens = _tokenize(expr)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def eat(expect: str | None = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("表达式意外结束")
        tok = tokens[pos]
        if expect is not None and tok != expect:
            raise ParseError(f"期望 {expect!r}, 得到 {tok!r}")
        pos += 1
        return tok

    def parse_arg():
        tok = peek()
        if tok is not None and tok.isdigit():
            return int(eat())
        return parse_expr()

    def parse_expr() -> Node:
        tok = peek()
        if tok is None or not tok[0].isalpha() and tok[0] != "_":
            raise ParseError(f"期望算子或终端，得到 {tok!r}")
        name = eat()
        if peek() != "(":  # 无括号 -> 终端
            if name not in terminals:
                raise ParseError(f"未知终端 {name!r} (合法：{terminals})")
            return Node(name=name, arity=0, value=name)
        eat("(")  # 算子调用
        args = [parse_arg()]
        while peek() == ",":
            eat(",")
            args.append(parse_arg())
        eat(")")
        return _build_op(name=name, args=args)

    node = parse_expr()
    if pos != len(tokens):
        raise ParseError(f"表达式有多余内容: {tokens[pos:]}")
    return node


if __name__ == "__main__":
    # 自测：纯语法层，不碰数据/API
    t = parse("div(rank(amount), rank(open))")
    assert str(t) == "div(rank(amount), rank(open))"  # 无窗口算子可完整还原
    t = parse("ts_mean(close, 10)")
    assert t.name == "ts_mean" and t.value == 10 and t.children[0].value == "close"
    t = parse("ts_corr(scale(amount), high, 20)")
    assert t.name == "ts_corr" and t.value == 20 and len(t.children) == 2
    print("合法用例 ->", parse("add(ts_std(volume, 5), neg(low))"))

    for bad in ["foo(close)", "add(close)", "ts_mean(close)", "rank(close, 5)", "div(close,)"]:
        try:
            parse(bad)
        except ParseError:
            continue
        raise AssertionError(f"应抛 ParseError 却没抛：{bad}")

    print("✅ 解析层自测通过：合法表达式建树正确、非法表达式全部拦截")
