# TIMELINE

## 2026-06-11

- **强化学习挖因子（AlphaGen 范式）搭建**：转 RL 生成器——把因子表达式拆成后缀(RPN) token 序列，策略网络逐 token 生成，IC 当 reward。四文件 `src/rl/`：[`tokens.py`](../src/rl/tokens.py)（词表 46 + RPN↔Node + 合法性，语法层纯 Python）、[`env.py`](../src/rl/env.py)（gym 式 MDP，预算 mask 保证 `remaining≥栈深` 恒有合法动作，END 算同 regime 训练段 \|IC\| 奖励）、[`policy.py`](../src/rl/policy.py)（GRU 策略 + rollout + logprob/熵）、[`train_rl.py`](../src/rl/train_rl.py)（REINFORCE + 优势归一化 + 熵奖励，top 因子补 test_ic 存成 factor_library 同构 schema、直接喂 screen/deflated 闭环）。
- **踩坑修复**：① `for name in "ts_corr"` 逐字符遍历造垃圾 token；② `kind=="kind"` END 判定写错致栈深 +1；③ **`float("-inf")` mask → NaN 梯度**（`0*-inf`，nan_to_num 只抹前向、反向仍 nan，污染权重几轮后崩）→ 改 `-1e9` + `clip_grad_norm_`。
- **状态**：四件套齐全、各自自测绿、NaN 已修；正式冒烟待服务器 GPU 空闲（卡全满）。**尚无 IC 上升实测**。诚实限定：瓶颈在 CPU（奖励评估 pandas）非 GPU、REINFORCE v1（PPO 待升级）、成败仍看 regime rho。笔记 [`强化学习挖因子.md`](./强化学习挖因子.md)。
- **顶层重构：抽出 core（分支 `refactor/extract-core`）**：多生成器（GP/RL/未来 LLM）都直接 import `src.gp` 的味道——诊断为「共享核被错放在 gp 下」。把 `tree(Node)/operators/evaluator` 三件套 `git mv` 到 [`src/core/`](../src/core/)（中立核），GP/RL/fitness/全 scripts 共 24 处 import 改向 core。验证：无残留、全编译、ruff 全过、依赖链通（仅本地缺 torch/joblib，服务器无碍）。确立**三层 + 两契约**架构（生成器平级只依赖 core+data、互不依赖；因子=`Node`、产出=`factor_library` schema、消费者只认 schema），写成 [`顶层设计.md`](./顶层设计.md) 作为加新生成器的北极星。刻意不做：抽象基类/注册表（未到 rule of three）、统一 walkforward 旧 schema、`src/generators/` 子目录。
- 待办：服务器冒烟（GP 一轮 + RL 一轮）确认 import 改向无误 → 合并回 develop。

## 2026-06-10

- **因子动物园（生成/筛选解耦）**：按 mentor 方向重构——生成与筛选拆开。新增 [`generate_factors.py`](../scripts/generate_factors.py)（大批量生成树+去重+并行算 train/test **mean IC**，存 `factor_library.pkl`/`.csv`，指标改 IC 不用 ICIR）与 [`screen_factors.py`](../scripts/screen_factors.py)（按 |test IC| 取 top-K + 测试段回测）。脚本语法/依赖/签名已核验。纪律：测试集选 top = 选择偏差，winner 仍要过 deflated。这套是后续 RL 挖因子（PPO 换 GP、IC 当 reward）的底座。笔记 [`因子动物园.md`](./因子动物园.md)。
- 待办：服务器冒烟（`--n 200`）→ 看 test IC 尾部 → 放量 5000 → winner 过 deflated。
- **Regime 断点（本阶段最硬发现）**：因子动物园放量 5000 后，naive 按 test IC 选翻车（train_ic≈0.01/test_ic≈0.07，选择偏差）。查根因算全库 train/test IC 秩相关 rho=**−0.295**（负！），再做逐年 IC 持续性诊断（[`diagnose_persistence.py`](../scripts/diagnose_persistence.py)）——年×年矩阵暴露**两个 regime 簇 {2019,2020} vs {2021–2024}，断点在 2020→2021，跨簇全负、簇内 +0.7~0.85**。这解释了之前所有 walk-forward OOS 崩塌：训练窗全骑在断点上、混了两个相反 regime。**修法**：训练段砍到 2021 起（`--train-start`，避开断点），rho **−0.295→+0.801**，top 因子 train/test 同号同量级、测试段 long-only T+1 sharpe 0.6~0.9。**winner `div(log(rank(amount)), neg(rank(open)))` 三冠齐全**：|NW_t|=4.22 > E[max t]=3.41（N=1730）、deflated p=0.0408——**项目第一个在完整试验数下扛过 deflated 的因子**（之前只在小 N 下过）。结论：**瓶颈不是没信号，是训练窗骑 regime 断点；同 regime 内纯量价因子强持续可迁移**。限定：amount 家族主导、regime 条件性、压线过非碾压。笔记 [`regime断点.md`](./regime断点.md)。

## 2026-06-06

- **适应度函数纵深 B+·① 时序稳定性**：`make_objective` 改为切 K=3 子区间各算 ICIR，适应度取 `|mean(ICIR_k)| − λ·std(ICIR_k)`，逼搜索偏向跨期稳定因子。结果：fold2 NW_t 从 1.29 → **2.84**（p=0.0045 ✓）、fold3 NW_t 2.23 → 2.59（p=0.0097 ✓），信号质量大幅提升，**首次在小 N（<30）下通过 deflated Sharpe**。
- **新发现：稳定性适应度的方向盲区**：`abs(arr.mean())` 对正负 ICIR 一视同仁，选出了两个一致负方向因子（ICIR=-0.29）；回测多头买顶分位 → 实际买了预测跑输的票 → 三折全负 sharpe。修法：`run_backtest.py` 在喂入回测前按 `train_icir` 符号翻转因子。负 ICIR 因子本身是有效反转信号，并非无效，只需方向修正。
- **结论**：稳定性适应度解决了正确的问题（OOS 信号更真实），OOS |ICIR| 均值 0.16→0.21，但暴露了回测方向 bug。两个独立问题被同一次实验同时暴露。
- **适应度函数纵深 B+·② 正交性压力**：适应度叠加 `−ORTH_LAMBDA·|corr(factor, amount)|` 逼搜索逃离流动性吸引盆。结果：**惩罚口径打偏**——只挡得住 amount 的单调变换（volume/ts_std(amount)），挡不住「以 amount 为输入的关系类因子」`ts_corr(amount, high)`（其值是相关系数，与 amount 水平不单调，从缝里钻出）。真发现：正交化的惩罚量纲必须对准要剥离的形态。副产物：② 反选出统计质量最高的因子，fold2 NW_t **3.40**（p=0.0007），首次稳过小 N deflated Sharpe。
- **工程：并行评估 + 统一评估入口**：`_eval_objs` 用 joblib（`n_jobs` 从 config 读，公共服务器占 10 核）并行适应度评估 + 预算 fwd 收益排名常量，全量 walk-forward 从 **8h 压到 1h**。新增 [`evaluate_run.py`](../scripts/evaluate_run.py)：一个 pkl 进、数据只读一次、每折树只求值一次，一次跑完「方向修正 → 测试段回测三层 → NW 显著性 → deflated 存活线」，消除原四脚本各自重复读数据/重算树。
- **B+ 收口结论**：纯量价 GP 能挖到统计显著的真信号（① 前置稳定性 + ② 选出 `ts_corr(amount,high)`，首次过 deflated），但方向修正后实盘多头在 2022 熊市仍亏（截面排序 ≠ 多头绝对 PnL，D 节复现）。笔记 [`适应度函数纵深.md`](./适应度函数纵深.md)。
- **多因子合成（成果）**：rank 归一 + 符号对齐 + 等权合成跨折候选（[`synthesize.py`](../scripts/synthesize.py)）。复合因子 ICIR 0.254 / NW_t 5.81 超过任一单因子、分组完美单调（G1 11.9→G5 34.2bp）。毛多空 sharpe **0.75**、年化 11%，但被 10bp 换手成本吃到净≈0。**5 日平滑（低换手实现）后净多空 sharpe 转正 0.38、年化 5%、ICIR 几乎无损，且对窗口稳健（w=3–10 同量级）**。收口结论：**纯量价 alpha 在 A 股的瓶颈是换手成本结构，而非信号匮乏**——首次给出有数据、有方法、有正向落地的成果。笔记 [`多因子合成.md`](./多因子合成.md)。
- **现实约束回测（D）**：`simple.py` 加 `_tradable_masks`（涨跌停/停牌过滤）+ long-only 口径；runner [`run_backtest.py`](../scripts/run_backtest.py) 做「纸面多空 → 现实多头·收盘 → 现实多头·T+1 开盘」三级瀑布（T+1 用 open[t+1+p]/open[t+1]-1 成交收益）。
- **核心发现**：现实约束把因子**重排序**——OOS ICIR 最强的 fold3 实盘只排第二，ICIR 最低的 fold2（低波因子）才是唯一扛过三层约束的（T+1 sharpe 0.557、最大回撤 −7%），但仍未过显著性。**ICIR 最强 ≠ 实盘最好**。
- **笔记**：D 节收口 [`回测现实性.md`](./回测现实性.md)。
- （均在 `develop` 分支开发，待合并入 `main`）

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
