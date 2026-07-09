# TIMELINE

## 2026-07-09

- **AlphaGen 正式档（pool_capacity=20, steps=250000）在服务器跑通，11 小时**：
  `InvalidExpressionException` 防御补丁扛住了全程（`eval_cnt` 到 19123，没崩）。结果：
  train pool IC 单调爬到 0.186，OOS test IC 全程在 0.05~0.12 徘徊、末期还比中期略降
  （0.064→0.054）——train/test 分道扬镳，是很干净的过拟合信号，跟 GP/RL/LLM 三个自研
  生成器已经实锤的"纯量价搜索撞 amount 天花板"结论一致，这次是拿论文原版算法印证的
  第四份独立证据。
- **`bridge_alphagen.py` 首次真实桥接：覆盖率 0/20**——因子池里数值常量、
  Greater/Sum/Med/Mad 这几个到处出现，本项目 `operators.py`/`Node` 体系原来都不支持，
  一个都翻译不了。补齐：① `src/core/tree.py`/`evaluator.py` 加数值常量叶子节点支持
  （`Node(value=<float>)` 广播成常数宽表）；② `src/core/operators.py` 新增
  greater/less/ts_sum/ts_median/ts_var/ts_mad/ts_cov 七个算子（**这是 GP/RL/LLM 共用的
  核心算子表，词表一并变大，不只是修桥接**）；③ `$vwap` 翻译成 `div(amount, volume)`
  组合表达式而不是判不可翻译。用真实池子里失败的 11 条表达式复测：0/11 → 9/11（只剩
  WMA/EMA 两个依然不支持，`Sign` 是本项目之前专门删掉的 reward-hacking 温床，刻意不
  重新引入）。顺手修了 `alphagen_bridge.py` 自测里一处过时断言（`Node.__str__` 不显示
  时序窗口是已知问题，2026-07-04 记过，这次只改测试期望值对齐现状，没有动 `__str__`
  本身——那是更大范围的 dedup key 决定，留给后续）。详见
  [AlphaGen复现.md](./AlphaGen复现.md)。

## 2026-07-07

- **AlphaGen 复现路线转折：放弃 qlib/baostock，改喂本项目自己的 tushare 数据**。服务器上
  qlib/baostock 数据链路连续踩坑（baostock 私有协议端口 10030 被计算节点防火墙挡死、qlib
  官方数据源 Azure blob 被禁 409、PyPI 上叫"qlib"的其实是 2018 年无关废弃包、真正的
  `pyqlib` 装最新版又跟锁定的老 numpy ABI 不兼容），其中网络层的坑无法通过改代码绕开。
  改路线：`Expression.evaluate()` 只依赖鸭子类型接口不依赖真 qlib，新增
  `alphagen_generic/tushare_data.py::TushareStockData` 顶替官方 `StockData`，直接从
  `data/cache/prices_clean.parquet` 构造；`scripts/rl.py` 训练/测试 segments 改成跟本项目
  GP/RL/LLM/QuantFactor 同一套 train_start=2021-01-01 起始约定。本机合成数据把整条链路
  （PPO 训练+因子池+checkpoint 落盘）跑通、退出码 0。不再有独立的"官方 CSI300 口径"，全部
  跟本项目其他生成器同一份数据、天然可比；`scripts/bridge_alphagen.py` 桥接脚本仍需要跑
  （AlphaGen 落盘的还是它自己的表达式语法，需转成本项目 `Node` 才能进
  `compare_generators.py`）。详细踩坑记录与依赖变化表见
  [AlphaGen复现.md](./AlphaGen复现.md)。

## 2026-07-04

- **mentor 新方向：文献复现 → 论文起点**。mentor 要求先调研现有因子挖掘方法、挑几个复现，
  作为论文的起点，后续再谈改进。调研落地为两个**发表过的** RL 系 baseline，都跟自研 GP/RL/LLM
  三生成器（见 [强化学习挖因子](./强化学习挖因子.md)、[LLM进化挖因子](./LLM进化挖因子.md)）放
  一起横向对比，同时刻意排除新的 LLM 系论文（temperature 不可复现 + API 不确定性太大，不利于
  "复现"这个目标）。
- **QuantFactor REINFORCE（arXiv:2409.05144，无官方代码，照论文公式实现）**：新增
  `src/rl/alpha_pool.py`（因子池，梯度下降拟合线性组合权重）、`src/rl/pool_env.py`（组合 IC +
  IR 时变阈值塑形奖励）、`policy.py::greedy_rollout`（贪婪解码，供贪婪基线用）、
  `scripts/run_quantfactor.py`（每 episode 双轨迹：采样+贪婪，advantage=两者 reward 差，不再用
  batch 内归一化）。本机合成数据冒烟跑通，`compare_generators.py` 打通。笔记
  [QuantFactorREINFORCE复现.md](./QuantFactorREINFORCE复现.md)。
- **AlphaGen（KDD 2023，官方仓库 RL-MLDM/alphagen，走真跑代码路径）**：clone 到
  `baselines/alphagen/`（不进 `src/`，外部产物不算生成器）。本机 macOS arm64 装官方
  `requirements.txt` 踩坑——`numpy==1.20.1` 等老版本无 arm64 wheel + 新版 setuptools 缺
  `pkg_resources`，装不上；放宽版本装出独立环境 `alphagen-repro`（仅本机验证代码/环境用，
  服务器 Linux+CUDA 应直接用官方原始锁定版本）。新增 `src/rl/alphagen_bridge.py`（表达式字符串
  →`Node` 翻译器，算子映射表 + 不可翻译判定）与 `scripts/bridge_alphagen.py`（因子池 json → 自己
  数据重新算 IC → factor_library.pkl），假数据验证链路通。**真实训练数据在实验室服务器上，本次
  只做到环境验证+桥接管线就绪，正式跑通与结果留给服务器**（README 写了跑法）。笔记
  [AlphaGen复现.md](./AlphaGen复现.md)。
- **顺手修的两个既有 bug**：① `src/core/operators.py::ts_max` 实际调用的是 `.rolling().min()`
  （应为 `.max()`，copy-paste 遗留），影响所有历史生成器，已修。② 发现 `Node.__str__`
  （`src/core/tree.py`）不显示时序窗口，会把不同窗口的因子（如 `ts_mean(close,5)` 和
  `ts_mean(close,20)`）误判成同一个 dedup key——`src/llm/clean.py` 早前已踩过这个坑并修过
  （`to_expr`），但 `train_rl.py`/`generate_factors.py` 目前仍是裸 `str(tree)`，同样风险仍在，
  这次只在新写的 `pool_env.py` 里避坑（本地 `_to_expr`，不跨生成器 import），没有回头改旧代码，
  如实记录、留给后续判断要不要修。

## 2026-06-11

- **强化学习挖因子（AlphaGen 范式）搭建**：转 RL 生成器——把因子表达式拆成后缀(RPN) token 序列，策略网络逐 token 生成，IC 当 reward。四文件 `src/rl/`：[`tokens.py`](../src/rl/tokens.py)（词表 46 + RPN↔Node + 合法性，语法层纯 Python）、[`env.py`](../src/rl/env.py)（gym 式 MDP，预算 mask 保证 `remaining≥栈深` 恒有合法动作，END 算同 regime 训练段 \|IC\| 奖励）、[`policy.py`](../src/rl/policy.py)（GRU 策略 + rollout + logprob/熵）、[`train_rl.py`](../src/rl/train_rl.py)（REINFORCE + 优势归一化 + 熵奖励，top 因子补 test_ic 存成 factor_library 同构 schema、直接喂 screen/deflated 闭环）。
- **踩坑修复**：① `for name in "ts_corr"` 逐字符遍历造垃圾 token；② `kind=="kind"` END 判定写错致栈深 +1；③ **`float("-inf")` mask → NaN 梯度**（`0*-inf`，nan_to_num 只抹前向、反向仍 nan，污染权重几轮后崩）→ 改 `-1e9` + `clip_grad_norm_`。
- **状态**：四件套齐全、各自自测绿、NaN 已修。笔记 [`强化学习挖因子.md`](./强化学习挖因子.md)。
- **RL 全量实测 + 收口（CPU）**：① 工程补全——超参进 `configs/rl:` 段、**全存**探索因子（topk=0）+ `n_explored` 真实试验数、joblib 并行落盘。② 第一轮（max_len 20、无 parsimony）RL **学会靠 bloat 过拟合**：探索 18124 个、top 全 size-19、train IC 0.086（>GP 0.054）但 #1 train+0.086/test−0.073 翻号，平均奖励全程不爬，sharpe 77% 是集中度 artifact（NW_t 1.14）。③ 修法 `奖励=|IC|−parsimony·size`(0.001)+`max_len=10` 进 config。④ 第二轮 bloat 治住：size 7-9、rho **+0.856**、top 不翻号——**但 #1 仍没过 deflated**（N=7262→E[max t]3.78 > NW_t 2.58）。⑤ **对比随机基线（因子动物园 N=1730 winner NW_t 4.22 ✓）：RL 搜 4× 多、撞同一 amount 天花板、winner 反更弱、没过**。两条硬结论：**生成器不是瓶颈（GP/随机/RL 同撞 amount 顶，特征才是瓶颈）；搜索有税（搜越多 deflated 门槛越高）**。RL 定位为**可复现 baseline**，供后续方法对比。
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
