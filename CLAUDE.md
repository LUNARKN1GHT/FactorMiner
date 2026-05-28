# FactorMiner - 项目约定

## 项目概述

A 股因子挖掘实验项目，基于遗传规划 (GP) 算法自动搜索 alpha 因子表达式。实验室研究用途。

## 技术栈

- Python 3.11+
- 核心依赖：numpy, pandas, deap, scipy
- 数据源：akshare（A 股免费行情接口，待对接）
- 代码规范：ruff（lint + format）

## 项目结构

```text
src/
├── data/          # 数据获取与处理
│   └── loader.py  # 行情数据加载接口（TODO: 对接 akshare）
├── gp/            # 遗传规划引擎
│   ├── operators.py  # 因子算子（算术、截面、时序）
│   ├── tree.py       # 表达式树结构
│   └── engine.py     # GP 进化主循环
├── evaluation/    # 因子评估
│   └── metrics.py    # IC、ICIR、分组收益
└── backtest/      # 简单回测
    └── simple.py     # 分组多空回测

configs/           # YAML 实验配置
scripts/           # 运行脚本入口
notebooks/         # Jupyter 实验笔记
data/              # 本地数据（不入库）
results/           # 实验结果（不入库）
```

## 开发规范

- 代码格式化 & lint: `ruff check .` / `ruff format .`
- 配置文件: `ruff.toml`
- 运行入口: `python scripts/run_gp.py [config_path]`

## 当前状态

- 骨架代码已搭建，GP 引擎核心逻辑已实现
- 数据接口尚未对接（loader.py 中为 NotImplementedError）
- 下一步：对接数据源 → 实现适应度函数 → 运行首次进化实验
