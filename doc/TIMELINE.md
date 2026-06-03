# TIMELINE

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
