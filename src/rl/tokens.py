"""RL 因子挖掘·语法层：token 词表 + RPN(后缀)序列 ↔ Node 树 + 合法性判定。

把因子表达式拆成 token 序列，策略网络逐 token 生成。用后缀(RPN)：
操作数压栈、算子弹出其元数压回结果——合法性 = 栈纪律。
时序算子的窗口烤进 token 名（ts_mean_5/10/20），每个 token 元数+窗口都固定。
本层不依赖数据/torch，可独立单测。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.gp.tree import Node

# 与数据对齐的核心终端
CORE_TERMINALS = ["open", "high", "low", "close", "volume", "amount"]
TS_WINDOWS = (5, 10, 20)


@dataclass(frozen=True)
class Token:
    """词表里的一个 token"""

    name: str
    """Token 名"""

    kind: str
    """Token 类型"""

    arity: int
    """建树时的弹栈个数"""

    op_name: str | None = None
    """对应 operators 注册表里的算子名"""

    window: int | None = None
    """时许算子窗口，建树时存进 Node.value"""


def build_vocab(
    terminals: list[str] | None = None, windows: tuple[int, ...] = TS_WINDOWS
) -> list[Token]:
    """构造完整词表。顺序固定——策略网络的输出维度按此表顺序对齐"""
    terminals = terminals or CORE_TERMINALS
    vocab: list[Token] = []

    # 终端
    for t in terminals:
        vocab.append(Token(t, "terminal", 0))

    # 二元算数
    for name in ("add", "sub", "mul", "div"):
        vocab.append(Token(name=name, kind="op", arity=2, op_name=name))

    # 一元算数
    for name in ("neg", "abs", "log", "rank", "scale"):
        vocab.append(Token(name=name, kind="op", arity=1, op_name=name))

    # 一元时许
    for name in (
        "ts_mean",
        "ts_std",
        "ts_rank",
        "delay",
        "delta",
        "ts_max",
        "ts_min",
        "ts_argmax",
        "decay_linear",
    ):
        for w in windows:
            vocab.append(Token(name=f"{name}_{w}", kind="op", arity=1, op_name=name, window=w))

    # 二元时序
    for name in ("ts_corr",):
        for w in windows:
            vocab.append(Token(name=f"{name}_{w}", kind="op", arity=2, op_name=name, window=w))

    # 终止
    vocab.append(Token(name="END", kind="end", arity=0))
    return vocab


def is_legal_next(token: Token, stack_depth: int, length: int, min_len: int) -> bool:
    """给定当前生长长度，判断这个 token 是否合法

    非法的 token 将在策略里呗压缩成 0 概率
    """
    if token.kind == "terminal":
        return True
    if token.kind == "op":
        return stack_depth >= token.arity
    return stack_depth == 1 and length >= min_len


def rpn_to_node(tokens: list[Token]) -> Node:
    """把一条 RPN token 序列构建成 Node 树。非法序列抛出 ValueError"""
    stack: list[Node] = []
    for token in tokens:
        if token.kind == "end":
            break
        if token.kind == "terminal":
            stack.append(Node(name=token.name, arity=0, value=token.name))
        else:
            if len(stack) < token.arity:
                raise ValueError(
                    f"非法 RPN：'{token.name}' 需要 {token.arity} 个操作数，栈深仅 {len(stack)}"
                )
            children = [stack.pop() for _ in range(token.arity)][::-1]
            stack.append(
                Node(name=token.op_name, arity=token.arity, children=children, value=token.window)
            )
    if len(stack) != 1:
        raise ValueError(f"非法 RPN：结束时栈深 = {len(stack)}, 应为 1")
    return stack[0]


if __name__ == "__main__":
    # 自测：不需要数据，只验语法层
    vocab = build_vocab()
    name2tok = {t.name: t for t in vocab}
    print(f"词表大小 = {len(vocab)}")

    # 手搓一条 RPN：amount rank open rank div  →  div(rank(amount), rank(open))
    seq = [name2tok[n] for n in ("amount", "rank", "open", "rank", "div", "END")]
    tree = rpn_to_node(seq)
    print("RPN→Node:", tree)
    assert str(tree) == "div(rank(amount), rank(open))", str(tree)

    # 带窗口的时序：close ts_mean_10 END  →  ts_mean(close, 10)
    seq2 = [name2tok[n] for n in ("close", "ts_mean_10", "END")]
    tree2 = rpn_to_node(seq2)
    print("RPN→Node:", tree2, "| window(value) =", tree2.value)
    assert tree2.value == 10

    print("✅ 语法层自测通过")
