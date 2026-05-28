"""GP 算子定义：用于构建因子表达式的基本运算。"""

import numpy as np
import pandas as pd

# ---- 二元算子 ----


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(y) > 1e-10, x / y, 0.0)
    return result


# ---- 一元算子 ----


def neg(x: np.ndarray) -> np.ndarray:
    return -x


def abs_op(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def log_op(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 1e-10, np.log(x), 0.0)


def rank(x: np.ndarray) -> np.ndarray:
    """截面排名，归一化到 [0, 1]。"""
    s = pd.Series(x)
    return s.rank(pct=True).values


# ---- 时序算子 ----


def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """滚动均值。"""
    return x.rolling(window, min_periods=1).mean()


def ts_std(x: pd.Series, window: int) -> pd.Series:
    """滚动标准差。"""
    return x.rolling(window, min_periods=1).std().fillna(0)


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """滚动排名（当前值在窗口中的分位数）。"""

    def _rank_pct(arr):
        s = pd.Series(arr)
        return s.rank(pct=True).iloc[-1]

    return x.rolling(window, min_periods=1).apply(_rank_pct, raw=False)


def delay(x: pd.Series, period: int) -> pd.Series:
    """时序延迟。"""
    return x.shift(period)


def delta(x: pd.Series, period: int) -> pd.Series:
    """时序差分。"""
    return x - x.shift(period)


# ---- 算子注册表 ----

BINARY_OPS = {
    "add": (add, 2),
    "sub": (sub, 2),
    "mul": (mul, 2),
    "div": (protected_div, 2),
}

UNARY_OPS = {
    "neg": (neg, 1),
    "abs": (abs_op, 1),
    "log": (log_op, 1),
    "rank": (rank, 1),
}

TS_OPS = {
    "ts_mean": (ts_mean, 2),
    "ts_std": (ts_std, 2),
    "ts_rank": (ts_rank, 2),
    "delay": (delay, 2),
    "delta": (delta, 2),
}
