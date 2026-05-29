"""实验日志模块。

每次实验自动创建带时间戳的目录：
  results/logs/exp_20240115_143022/
    run.log      - 人类可读的完整日志（同时输出到终端）
    stats.jsonl  - 每代进化指标，每行一个 JSON，便于后续分析

用法：
  from utils.logger import setup_experiment_logger, log_generation, load_stats

  logger, log_dir = setup_experiment_logger()
  log_generation(log_dir, gen=1, best_ic=0.05, mean_ic=0.03, best_expr="ts_mean(close,5)")

  # notebook 里读取分析
  stats = load_stats(log_dir)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

_LOG_ROOT = Path(__file__).parents[2] / "results" / "logs"


def setup_experiment_logger(
    exp_id: str | None = None,
    log_root: Path | str | None = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """初始化实验日志，返回 (logger, log_dir)。

    Parameters
    ----------
    exp_id:
        实验 ID，默认自动生成时间戳（exp_YYYYMMDD_HHMMSS）。
    log_root:
        日志根目录，默认 results/logs/。
    level:
        日志级别。

    Returns
    -------
    logger:
        已配置好的 Logger，同时输出到终端和文件。
    log_dir:
        本次实验的日志目录，传给 log_generation() 使用。
    """
    if exp_id is None:
        exp_id = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    log_root = Path(log_root) if log_root else _LOG_ROOT
    log_dir = log_root / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(exp_id)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("实验开始：%s", exp_id)
    logger.info("日志目录：%s", log_dir)

    return logger, log_dir


def log_generation(
    log_dir: Path,
    gen: int,
    best_ic: float,
    mean_ic: float,
    best_expr: str,
    extra: dict | None = None,
) -> None:
    """记录一代进化的关键指标到 stats.jsonl。

    Parameters
    ----------
    log_dir:
        由 setup_experiment_logger() 返回的实验目录。
    gen:
        当前代数。
    best_ic:
        本代最优个体的 IC 值。
    mean_ic:
        本代种群平均 IC。
    best_expr:
        本代最优因子表达式字符串。
    extra:
        其他需要记录的字段，如 {"elapsed": 12.3, "pop_size": 500}。
    """
    record: dict = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "gen": gen,
        "best_ic": round(best_ic, 6),
        "mean_ic": round(mean_ic, 6),
        "best_expr": best_expr,
    }
    if extra:
        record.update(extra)

    with open(log_dir / "stats.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_stats(log_dir: Path | str) -> list[dict]:
    """读取实验所有代数的统计记录，用于 notebook 分析。"""
    p = Path(log_dir) / "stats.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
