"""因子表达式算子定义：GP/RL/LLM/AlphaGen 桥接共用的算子注册表（`src/core` 是中立核，
见 `doc/顶层设计.md`）。2026-07-09 为了让 AlphaGen 桥接（`src/rl/alphagen_bridge.py`）覆盖率
从 0% 提上来，新增了 greater/less/ts_sum/ts_median/ts_var/ts_mad/ts_cov 这几个 AlphaGen
有而本项目原来没有的算子——**注册在这里意味着 GP/RL/LLM 的搜索空间也会一并变大**，不是只
影响桥接，如果要跟改动前的历史结果比较需要注意这一点。"""

import numpy as np
import pandas as pd

# ---- 二元算子 ----


def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x + y


def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x - y


def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x * y


def protected_div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    # 除法本身按格子对齐即可；唯一要防的是除零产生的 inf
    out = x / y
    # 把 ±inf 替换成 0；真正的 NaN（停牌/缺失）保留，交给后面 IC 计算时过滤
    return out.replace([np.inf, -np.inf], 0.0)


def greater(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """逐元素取较大值（AlphaGen 的 Greater，对齐语义，非截面/时序算子）"""
    return np.maximum(x, y)


def less(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """逐元素取较小值（AlphaGen 的 Less）"""
    return np.minimum(x, y)


# ---- 一元算子 ----


def neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


def abs_op(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)


def log_op(x: pd.DataFrame) -> pd.DataFrame:
    # 只对正数取 log；非正数先置 NaN（log 无定义），结果自然是 NaN
    # 不要像原来那样用 np.where —— 它会把 DataFrame 退化成 ndarray，丢掉行列索引
    return np.log(x.where(x > 1e-10))  # type: ignore


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面排名"""
    # 截面排名：关键是 axis=1 —— 对「每一行(每一天)」内部、跨股票排名
    # 原来的 pd.DataFrame(x).rank() 把整张表拉平排名，是错的
    return x.rank(axis=1, pct=True)  # type: ignore


def scale(x: pd.DataFrame) -> pd.DataFrame:
    """截面缩放：每天除以当日 ｜值｜ 之和，使绝对值和为 1"""
    denom = x.abs().sum(axis=1)
    return x.div(denom, axis=0).replace([np.inf, -np.inf], 0.0)


# ---- 时序算子 ----


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动均值。"""
    return x.rolling(window, min_periods=1).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动标准差。"""
    return x.rolling(window, min_periods=1).std().fillna(0)


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动排名（当前值在窗口中的分位数）。"""

    def _last_rank(arr):
        return (arr <= arr[-1]).mean()

    return x.rolling(window, min_periods=1).apply(_last_rank, raw=True)


def delay(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """时序延迟。"""
    return x.shift(period)


def delta(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """时序差分。"""
    return x - x.shift(period)


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最大值"""
    return x.rolling(window=window, min_periods=1).max()


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口最小值。"""
    return x.rolling(window, min_periods=1).min()


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """距窗口内最高点过了几天，是一种择时/位置信号"""

    def _days_since_high(arr: pd.Series):
        return len(arr) - 1 - arr.argmax()

    return x.rolling(window=window, min_periods=1).apply(_days_since_high, raw=True)


def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性加权平均：越近的日子权重越大，之后进行归一化"""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    out = x.shift(window - 1) * weights[0]
    for i in range(1, window):
        out = out + x.shift(window - 1 - i) * weights[i]
    return out


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口求和（AlphaGen 的 Sum）"""
    return x.rolling(window, min_periods=1).sum()


def ts_median(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口中位数（AlphaGen 的 Med）"""
    return x.rolling(window, min_periods=1).median()


def ts_var(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口方差（AlphaGen 的 Var）"""
    return x.rolling(window, min_periods=1).var().fillna(0)


def ts_mad(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口平均绝对离差：跟 AlphaGen 的 Mad 对齐——每个窗口内减掉的是
    「该窗口自己的均值」，不是外部另算的一条滚动均值序列，所以用 apply 而不是
    两次独立 rolling 相减（那样会对错参照点）"""

    def _mad(arr):
        return np.mean(np.abs(arr - arr.mean()))

    return x.rolling(window, min_periods=1).apply(_mad, raw=True)


# ---- 时序二元算子 ----


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """两序列的滚动时序相关系数"""
    out = x.rolling(window=window, min_periods=max(2, window // 2)).corr(y)
    return out.replace([np.inf, -np.inf], 0.0)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """两序列的滚动协方差（AlphaGen 的 Cov）"""
    return x.rolling(window=window, min_periods=max(2, window // 2)).cov(y)


# ---- 算子注册表 ----

BINARY_OPS = {
    "add": (add, 2),
    "sub": (sub, 2),
    "mul": (mul, 2),
    "div": (protected_div, 2),
    "greater": (greater, 2),
    "less": (less, 2),
}

UNARY_OPS = {
    "neg": (neg, 1),
    "abs": (abs_op, 1),
    "log": (log_op, 1),
    "rank": (rank, 1),
    "scale": (scale, 1),
}

TS_OPS = {
    "ts_mean": (ts_mean, 2),
    "ts_std": (ts_std, 2),
    "ts_rank": (ts_rank, 2),
    "delay": (delay, 2),
    "delta": (delta, 2),
    "ts_max": (ts_max, 2),
    "ts_min": (ts_min, 2),
    "ts_argmax": (ts_argmax, 2),
    "decay_linear": (decay_linear, 2),
    "ts_sum": (ts_sum, 2),
    "ts_median": (ts_median, 2),
    "ts_var": (ts_var, 2),
    "ts_mad": (ts_mad, 2),
}

TS_BINARY_OPS = {
    "ts_corr": (ts_corr, 3),
    "ts_cov": (ts_cov, 3),
}
