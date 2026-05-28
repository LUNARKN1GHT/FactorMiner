# FactorMiner

基于遗传规划 (Genetic Programming) 的 A 股 alpha 因子自动挖掘。

## 目标

利用 GP 算法从 A 股行情数据中自动搜索有效的 alpha 因子表达式，复现量化因子挖掘的经典方法。

## 方法

1. **表达式树表示**：用树结构表达因子公式，叶节点为行情特征（open/close/volume 等），内部节点为算子
2. **算子集**：算术运算、截面排名、时序统计（ts_mean, ts_std, delay, delta 等）
3. **进化搜索**：通过锦标赛选择、子树交叉、子树变异迭代优化因子表达式
4. **适应度评估**：以 Rank IC (Spearman 相关系数) 作为主要适应度指标

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 GP 因子挖掘（待数据对接后可用）
python scripts/run_gp.py

# 指定配置
python scripts/run_gp.py configs/default.yaml
```

## 项目结构

```text
src/
├── data/          # 数据获取与处理
├── gp/            # GP 进化引擎（算子、表达式树、主循环）
├── evaluation/    # 因子评估（IC、ICIR、分组收益）
└── backtest/      # 分组回测

configs/           # 实验配置（YAML）
scripts/           # 运行脚本
notebooks/         # 实验笔记
```

## 进度

- [x] 项目骨架搭建
- [x] GP 引擎核心实现（表达式树、遗传操作、进化主循环）
- [x] 因子评估指标（IC、ICIR、分组收益）
- [x] 简单分组回测
- [ ] A 股数据接口对接
- [ ] 适应度函数实现
- [ ] 首次进化实验
