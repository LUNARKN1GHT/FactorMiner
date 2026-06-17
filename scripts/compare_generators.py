"""跨生成器对照：把多个 factor_library schema 的 pkl 拉成一张表，公平对比生成器。

用 N 无关指标（rho、winner |NW_t|、IC 天花板、迁移率、平均 size）横向比 GP / 随机动物园 / RL。
deflated 默认按各自探索数 N 报（搜得多门槛高、N 不同不直接可比）；--n 可统一 N 做公平判定。
只读 pkl + 数据；数据只读一次（假设所有库同数据集）。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr

from src.core.evaluator import evaluate, to_wide
from src.data.preprocess import to_panel
from src.evaluation.metrics import calc_ic_series

GAMMA = 0.5772156649015329  # 欧拉–马歇罗尼常数

_PKL_NAMES = ("factor_library.pkl", "rl_factors.pkl")  # 各生成器落盘的 pkl 文件名


def _is_library(path: Path) -> bool:
    """是不是 factor_library schema（dict 含 factors/config_path）。GP walkforward / 纯 list 不算。"""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return False
    return isinstance(obj, dict) and "factors" in obj and "config_path" in obj


def resolve_pkl(arg: str) -> str:
    """arg 是现成 pkl 路径就直接用；否则当生成器 tag（llm/rl/gp…），取最新一次 run 的 pkl。

    免得每次手敲 results/logs/exp_20260617_190222_llm/... 这种长时间戳路径。
    只认 factor_library schema 的 pkl，不兼容的（如 GP walkforward {folds,picks}）会明确报错。
    """
    p = Path(arg)
    if p.is_file():
        if not _is_library(p):
            raise SystemExit(f"{p} 不是 factor_library schema（dict 含 factors/config_path），不兼容对比")
        return str(p)
    runs = sorted(Path("results/logs").glob(f"*_{arg}"))  # 时间戳命名，字典序即时间序
    if not runs:
        raise SystemExit(f"找不到 pkl，也没有 tag={arg} 的 run（results/logs/*_{arg}）")
    latest = runs[-1]
    candidates = [latest / n for n in _PKL_NAMES if (latest / n).is_file()]
    candidates += [q for q in sorted(latest.glob("*.pkl")) if q not in candidates]
    for q in candidates:
        if _is_library(q):
            return str(q)
    raise SystemExit(f"{latest} 里没有 factor_library schema 的 pkl（tag={arg} 不兼容对比）")


def nw_tstat(ic: pd.Series, horizon: int) -> float:
    """重叠矫正后的 t 值（Bartlett 核，lag=horizon-1）。"""
    x = ic.to_numpy()
    n = len(x)
    mu = float(x.mean())
    dev = x - mu
    var = float(dev @ dev) / n
    for k in range(1, horizon):
        var += 2 * (1 - k / horizon) * float(dev[k:] @ dev[:-k]) / n
    se = (var / n) ** 0.5
    return mu / se if se > 0 else 0.0


def expected_max_t(n_trials: int) -> float:
    """N 次零假设检验中 t 最大值的期望。"""
    a = norm.ppf(1 - 1 / n_trials)
    b = norm.ppf(1 - 1 / (n_trials * np.e))
    return (1 - GAMMA) * a + GAMMA * b


def summarize(lib: dict, wide: dict, fwd: pd.DataFrame, method: str, horizon: int) -> dict:
    """一个因子库的 N 无关对比指标。"""
    facs = lib["factors"]
    split = pd.Timestamp(lib["split"])
    test_hi = fwd.index.max() + pd.Timedelta(days=1)
    n_search = lib.get("n_explored", len(facs))  # 真实探索数（RL 带，GP 退回库大小）

    tr = np.array([f["train_ic"] for f in facs], dtype=float)
    te = np.array([f["test_ic"] for f in facs], dtype=float)
    ok = ~(np.isnan(tr) | np.isnan(te))
    tr, te = tr[ok], te[ok]

    # winner = 按 |train_ic| 选（口径同 screen_factors），算其测试段 NW_t（N 无关的强度）
    win = max(facs, key=lambda r: abs(r["train_ic"]))
    ic = calc_ic_series(evaluate(win["tree"], wide), fwd, method=method)
    ic_te = ic.loc[(ic.index >= split) & (ic.index < test_hi)].dropna()

    return {
        "N": n_search,
        "rho": float(spearmanr(tr, te).correlation),  # 样本内 IC 预测样本外的能力
        "ceiling": float(np.max(np.abs(tr))),  # IC 天花板（最强样本内 |IC|）
        "win_nwt": abs(nw_tstat(ic_te, horizon)),  # winner 测试段 |NW_t|，N 无关
        "transfer": float(np.mean(np.sign(tr) == np.sign(te))),  # train/test 同号率
        "avg_size": float(np.mean([f["size"] for f in facs])),  # 平均树大小（查 bloat）
        "win_expr": win["expr"],
    }


def main(paths: list[str], common_n: int | None) -> None:
    raw = paths
    paths = [resolve_pkl(a) for a in paths]  # 支持简写 tag（llm/rl/gp）→ 最新 run 的 pkl
    for a, q in zip(raw, paths, strict=True):
        print(f"解析: {a} → {q}")
    with open(paths[0], "rb") as fh:
        first = pickle.load(fh)
    with open(first["config_path"]) as f:
        cfg = yaml.safe_load(f)
    method = cfg["evaluation"]["ic_method"]
    horizon = cfg["evaluation"].get("holding_period", 5)
    prices = pd.read_parquet(cfg["data"].get("clean_path", "data/cache/prices_clean.parquet"))
    wide = to_wide(prices.drop(columns=["fwd_ret"]))
    fwd = to_panel(prices, "fwd_ret")

    rows = []
    for p in paths:
        with open(p, "rb") as fh:
            lib = pickle.load(fh)
        s = summarize(lib, wide, fwd, method, horizon)
        s["run"] = Path(p).parent.name[-22:]
        s["bar"] = expected_max_t(common_n or s["N"])
        s["deflated"] = "✓" if s["win_nwt"] > s["bar"] else "✗"
        rows.append(s)

    hdr = (
        f"{'run':<24}{'N':>7}{'rho':>7}{'ceiling':>9}"
        f"{'win|NWt|':>10}{'E[maxt]':>9}{'def':>5}{'xfer':>6}{'size':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:<24}{r['N']:>7}{r['rho']:>+7.3f}{r['ceiling']:>9.4f}"
            f"{r['win_nwt']:>10.2f}{r['bar']:>9.2f}{r['deflated']:>5}"
            f"{r['transfer']:>6.2f}{r['avg_size']:>6.1f}"
        )
    print(
        "\n图例：ceiling=最强样本内|IC|  win|NWt|=winner 测试段 NW_t（N 无关的强度）  "
        "def=deflated  xfer=train/test 同号率  size=平均树大小"
    )
    if common_n:
        print(f"deflated 用统一 N={common_n} 判定（公平对比）。")
    else:
        print("deflated 按各自探索数 N；N 不同门槛不同 → 看 win|NWt| 才是 N 无关的公平比较。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pkls", nargs="+", help="pkl 路径，或生成器 tag（llm/rl/gp）= 取最新一次 run 的 pkl"
    )
    ap.add_argument("--n", type=int, default=None, help="统一 N 做 deflated 公平判定")
    args = ap.parse_args()
    main(args.pkls, args.n)
