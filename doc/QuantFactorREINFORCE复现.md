# QuantFactor REINFORCE 复现

> 论文：_QuantFactor REINFORCE: Mining Steady Formulaic Alpha Factors with
> Variance-bounded REINFORCE_（arXiv:2409.05144）。跟 [强化学习挖因子](./强化学习挖因子.md)
> 里自研的单因子 REINFORCE baseline 对照：同数据、同 tokens/env 语法层、同
> `FactorPolicy`（GRU）架构，只换训练算法与奖励结构。

## 为什么选这篇复现

调研阶段查过没有官方代码仓库（曾错查到 `microsoft/RD-Agent`，核实后那是另一个不相关的
多智能体量化框架）。既然没有仓库可以对照运行，"贴近原版"对这篇论文而言只能是**贴近论文
公式**——正好可以直接建在自己的 `src/rl/` 体系里，零新依赖，跟 AlphaGen（见
[AlphaGen复现](./AlphaGen复现.md)，走官方仓库路线）形成两条不同路径的对照。

## 论文的两个核心改动

### 1. 贪婪基线 REINFORCE（式 10），替代 AlphaGen 的 PPO

标准 REINFORCE 用 batch 内 reward 均值/方差归一化当 baseline（`train_rl.py` 的做法）。
这篇论文换了个思路：**每个 episode 跑两条轨迹**——一条策略网络随机采样，一条贪婪解码
（每步取 argmax），用贪婪轨迹的 reward 直接当 baseline：

```txt
g̃(θ) = (1/N) Σᵢ Σₜ sθ(a₁:ₜⁱ)·[r(a₁:ₜⁱ) − r(ā₁:ₜⁱ)]
```

`sθ` 是对数似然的梯度（score function），求和项其实就是采样轨迹的 `logprob_sum`；
reward 是轨迹级的（只在终止时非零），所以损失退化成
`-(r_sampled - r_greedy) * logprob_sum(采样轨迹)`。

好处：**不需要 value network**，比 PPO 省一次前向/一套 critic 训练，论文正是拿这个跟
AlphaGen 的 PPO 对比、主张"轨迹级奖励下 PPO 的 critic 是浪费"。

实现：`src/rl/policy.py::greedy_rollout()`，跟已有的 `rollout()` 结构完全一致，只把
`Categorical(...).sample()` 换成 `argmax`。训练循环（`scripts/run_quantfactor.py`）
每个 episode 调用两次（采样 + 贪婪），advantage = `sampled_reward - greedy_reward`，
**不再做 `train_rl.py` 那种 batch 内 reward 归一化**——这是这篇论文相对自家 baseline
的核心差异，别混用两套 baseline 逻辑。

### 2. 因子池 + IR 塑形奖励（式 13/14）

单因子自身 IC 不是奖励目标，奖励看的是**候选因子加入组合池后的表现**：

- **因子池**（`src/rl/alpha_pool.py::AlphaPool`）：维护固定容量的因子集合，线性组合
  权重通过梯度下降最小化 MSE `L(w) = mean((Σwᵢzᵢ − y)²)`；新因子随机初始化权重、
  跟池内已有因子一起联合优化；超容量时淘汰 `|weight|` 最小的一项。
  组合前对每个因子做**截面 rank 标准化**（映射到 `[-0.5, 0.5]`）——论文没有明说但
  不同量纲的因子直接线性组合没有意义，这一步是必须的。
- **IR 时变阈值塑形**（`src/rl/pool_env.py::PoolFactorEnv._reward`）：

  ```txt
  reward = combined_ic − λ·1{combined_ir ≤ clip[(t−α)·η, 0, δ]}
  ```

  训练早期（`t < α`）阈值钳在 0，几乎不罚低 IR 的因子，鼓励探索；训练后期阈值爬升到
  `δ`，逼策略在后段往更稳（IR 更高）的方向收敛。默认超参 `α=180000, η=1e-6,
δ=0.3, λ=0.02`（论文给的默认值，写进 `configs/default.yaml` 的 `quantfactor:` 段）。

## 跟自家 RL baseline 的关系

|             | `train_rl.py`（自研）     | `run_quantfactor.py`（本次复现）                                       |
| ----------- | ------------------------- | ---------------------------------------------------------------------- |
| 语法层/环境 | `tokens.py` + `FactorEnv` | 同一套 `tokens.py`，环境换 `PoolFactorEnv`                             |
| 策略网络    | `FactorPolicy`（GRU）     | 同一个类，不改架构                                                     |
| 奖励对象    | 单因子自身 `\|mean IC\|`  | 因子池组合 IC，按 IR 阈值塑形                                          |
| baseline    | batch 内 reward 归一化    | 贪婪轨迹 reward（式 10）                                               |
| 落盘 schema | `factor_library.pkl` 同构 | 同构（`config_path/split/n_explored/factors`），tag 改 `"quantfactor"` |

刻意保持"只换算法"：因子池实现（`AlphaPool`）只服务本 track，不与 AlphaGen 的池子共用——
两篇论文的池子拟合方式本就不同（AlphaGen 官方仓库另有一套拟合逻辑），共用会把"RL 算法
差异"和"池子拟合方式差异"两个变量混在一起，对比就不干净了。

## 顺手发现：`str(Node)` 丢时序窗口

写 `pool_env.py` 时发现：`Node.__str__`（`src/core/tree.py`）对非叶节点只拼
`children`，不拼 `value`（时序窗口）——`ts_mean(close,5)` 和 `ts_mean(close,20)`
会被打印成同一个字符串 `ts_mean(close)`。`src/llm/clean.py::to_expr` 已经踩过这个
坑并修过（那里的 `canon_key`/`to_expr` 注释直接写了"不能用 str(Node)"），但
`env.py`/`train_rl.py`（自研 RL baseline）和 `generate_factors.py` 的 `expr` 字段
目前仍是裸 `str(tree)`，存在同样的窗口坍缩风险（`best_reward[expr]` 去重可能把
不同窗口的因子当成同一个键，互相覆盖）。

本次新写的 `pool_env.py` 用了本地版 `_to_expr()`（渲染时把 `node.value` 追加进参数
列表）避免踩坑，没有跨生成器 import `src/llm/clean.py`（尊重"生成器互不依赖"的
架构约定，重复几行逻辑比引入耦合更合适）。**没有回头改 `train_rl.py`/
`generate_factors.py`**——那是已有生成器的既有行为，属于这次复现任务之外的东西，
如实记在这里，要不要修交给你和 mentor 判断。

## 已知局限 / 性能注意

- `AlphaPool._refit` 每次候选入池都要做一次梯度下降（默认 200 步），且每步是
  `(n_obs, k)` 的矩阵乘法，`n_obs` = 训练段有效 (date, code) 观测数（可能上万）。
  训练循环里这个函数每个 episode 都调用两次（采样+贪婪各一次），iters 一高会成为
  明显瓶颈——真跑全量前应先测单次 `_refit` 耗时，必要时降 `pool_gd_steps` 或对
  `n_obs` 做子采样。
- 没有实现 parsimony 惩罚：论文原设计没有这项，靠 `max_len` 硬卡树大小和 IR 阈值
  控质量。如果实测发现 bloat，再考虑要不要额外加（照论文原样先跑一遍看结果）。

## 冒烟测试

本机没有真实 `prices_clean.parquet`（数据在实验室服务器），用合成数据验证过管道：
`run_quantfactor.py` 在 `iters=4, batch=4, pool_capacity=3` 的极小配置下端到端跑通，
产出 `factor_library.pkl`（16 个因子，schema 与其他生成器一致），全库 train/test IC
秩相关能算出来、无 NaN/崩溃。正式规模的跑通与结果，留到服务器验证。
