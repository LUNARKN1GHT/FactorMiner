# TIMELINE

## 2026-06-05

- **算子扩展（C1+C2）**：`operators.py` 加 `ts_max/ts_min/ts_argmax/decay_linear/scale` 等即插即用算子；新增二元时序算子 `ts_corr`（「2 孩子 + 窗口」新节点形状，碰 operators/evaluator/engine/stgp 四处）。
- **reward hacking 修复**：`sign` 制造近常数因子把 ICIR 黑到 199（第三种 ICIR 不可信来源）。删 `sign` + 搜索目标加退化守卫（`nunique` 中位数下限）+ 汇总对 NaN 免疫。
- **C 扩算子结果**：walk-forward OOS 首次三项全面改善（均值 0.16、方差降、最差折翻三倍），主力是 `ts_corr`（价量关系）——「抬地板不抬天花板」；但仍过不了 deflated Sharpe。跨 run 对照脚本 [`compare_runs.py`](../scripts/compare_runs.py)。
- **C3 正交数据**：设计完成（baostock/基本面 + point-in-time 防泄露），但 tushare 接口限流、执行推迟。
- **笔记**：C 节收口 [`算子与数据.md`](./算子与数据.md)。下一站转 D（回测现实性）。
- （均在 `develop` 分支开发，待合并入 `main`）

## 2026-06-04

- **walk-forward 全量结果**：滚动窗三折跑通，整条前沿连表达式树落盘 `walkforward.pkl`，后续分析不重跑 GP。结论：因子方向稳但强度不稳（OOS \|ICIR\| 均值 0.14、标准差 0.10），幸存者清一色量/额类。
- **选择准则对比**：新增 [`compare_selection.py`](../scripts/compare_selection.py)，离线对比 train-max / min-size / parsimony——证明瓶颈不在选法，在搜出来的因子本就弱；顺带解开「负衰减」之谜（n=3 时一折 regime 运气主导汇总）。
- **显著性校正**：新增 [`significance.py`](../scripts/significance.py)（Newey-West 修单因子重叠，naive→NW 约 1.8× haircut）与 [`deflated.py`](../scripts/deflated.py)（Deflated Sharpe，break-even N\* 判活）。结论：**三折无一经得起重叠 + 选择偏差双重校正**。
- **笔记**：样本外验证完整结论写入 [`验证模块.md`](./验证模块.md)。
- **强类型 GP（STGP）**：新增 [`stgp.py`](../src/gp/stgp.py)（S/R 类型系统 + 类型化生成/交叉/变异 + recompute_type 自检），`Node` 加 `out_type`、`GPConfig` 加 `strongly_typed` 开关、`nsga2` 据开关选算子（排序逻辑复用）。配套 [`run_stgp.py`](../scripts/run_stgp.py) 与跨 run 对照 [`compare_runs.py`](../scripts/compare_runs.py)。结论：搜索空间剪干净（`非法 0 例`）但**样本外没改善**，钉死「瓶颈在特征贫瘠而非搜索合法性」。笔记 [`强类型GP.md`](./强类型GP.md)。
- （均在 `develop` 分支开发，待合并入 `main`）

## 2026-06-03

- **NSGA-II 多目标 GP**：实现多目标遗传规划 [`nsga2.py`](../src/gp/nsga2.py)，并配套入口脚本 [`run_gp_nsga2.py`](../scripts/run_gp_nsga2.py)。（在 `develop` 分支开发，待合并入 `main`）
- **训练窗口分割**：实现训练窗口分割 [`run_gp_walkforward.py`](../scripts/run_gp_walkforward.py)。直接完成实验脚本，
- **代码完善**：补充变量声明与 docstring，完善 `engine` / `fitness` / `operators` / `evaluator` 等函数定义。
- **笔记/实验**：更新优化模块笔记 [`优化模块.md`](./优化模块.md)，并在 [`05_gp_modules.ipynb`](../notebooks/05_gp_modules.ipynb) 中记录新实验。
- **问题**：
  - 在全量数据集上运行效率较低，参数优化还需要考虑。窗口过多可能也是导致计算速率下降的原因。

## 2026-06-02

- **GP 引擎优化**：
  - 初始化改用随机 half-and-half，修复 grow 模式的早停 bug。
  - 在适应度中加入树复杂度惩罚（配置项写入 [`default.yaml`](../configs/default.yaml)）。
- **笔记**：整理优化模块笔记 [`优化模块.md`](./优化模块.md)。
- **实验**：调参优化后得到新结果，记录在 [`05_gp_modules.ipynb`](../notebooks/05_gp_modules.ipynb)。

## 2026-06-01

- **GP 引擎**：增加树深度控制，避免表达式过深。
- **回测框架**：重写回测逻辑 [`simple.py`](../src/backtest/simple.py)，修复年化计算逻辑并处理无效数据。
- **实验分析**：在 [`04_results.ipynb`](../notebooks/04_results.ipynb) 中分析截断计算后的结果。

## 2026-05-31

- **适应度模块**：实现前向收益率计算与适应度函数 [`fitness.py`](../src/evaluation/fitness.py)，观察因子在未来能带来多少收益。
- **GP 链路打通**：[`run_gp.py`](../scripts/run_gp.py) 全链路跑通并通过测试，进化结束后保存 Top 因子。
- **工程化**：
  - 迁移到服务器上挂机计算。
  - `engine` 增加日志记录功能，便于实验回看。
  - 改用向量化计算，避免大批量重复计算。

## 2026-05-30

- **学习**：整理遗传规划学习笔记 [`遗传规划.md`](./遗传规划.md)。
- **求值器模块**：实现因子求值器 [`evaluator.py`](../src/gp/evaluator.py)，把表达式树求值成因子矩阵；同步调整 `operators` / `tree` / `engine` 的接口。

## 2026-05-29

- **数据模块**：
  - 设计好数据层，设置缓存并拉取数据。
  - 设计数据拉去模块。原本打算用 `akshare` 的，但是接口太不稳定，索性直接用个人的 `tushare` 接口调用了。
  - 完成数据预处理脚本 [`preprocess_data`](../scripts/preprocess_data.py)。对下载的数据进行预处理。
- **日志模块**：准备仓库的日志文件。方便后续挂机计算后回看结果。
