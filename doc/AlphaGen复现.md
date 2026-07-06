# AlphaGen 复现

> 论文：_Generating Synergistic Formulaic Alpha Collections via Reinforcement
> Learning_（KDD 2023 ADS track）。官方仓库 [RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen)。
> 跟 [QuantFactor REINFORCE 复现](./QuantFactorREINFORCE复现.md)（论文公式路径）不同，这篇走**官方仓库路径**——
> 直接跑他们的代码拿到论文口径的数字，再把挖出的表达式桥接回本项目的数据/评估流水线做横向对比。

## 为什么拆成两条路径

调研时确认 AlphaGen 有官方代码（QuantFactor REINFORCE 没有，见另一篇文档），"贴近原版"
对这篇而言就该是真的跑它的代码，而不是照论文重写一遍。但官方仓库跑在 qlib/baostock 的
CSI300 数据上，跟本项目的 tushare/hs300 是两套不同的数据管线——直接把官方数字和自己
GP/RL/LLM/QuantFactor 的数字放一张表比是不公平的（数据集都不同）。所以设计成两段：

1. **官方口径**：独立环境跑官方代码 + 官方数据，拿到论文本来的 IC/池子表现。
2. **桥接口径**：把官方挖出的因子表达式翻译成本项目 `Node`，在自己的数据上重新算 IC，
   这份数字才能跟 GP/RL/LLM/QuantFactor 放进同一张 `compare_generators.py` 表里。

两份数字都要留，不要混在一起充当一个数字。

## 环境隔离（本机 macOS arm64 踩坑记录）

新建独立 conda 环境 `alphagen-repro`（Python 3.9），仓库 clone 到 `baselines/alphagen/`
（项目根目录新增，不进 `src/`——这是"跑别人代码"，不是本项目的生成器，符合
[顶层设计](./顶层设计.md)"生成器只依赖 core+data"的原则，外部产物不算生成器）。

官方 `requirements.txt` 锁定的版本（`numpy==1.20.1, pandas==1.2.4, matplotlib==3.3.4`）
在 Apple Silicon 上没有官方 wheel，会触发源码编译，且现代 pip 的 build isolation 配合
新版 setuptools 缺 `pkg_resources`（`ModuleNotFoundError: No module named 'pkg_resources'`），
直接装原始锁定版本装不上。放宽到同大版本线内最早有 arm64 wheel 的版本
（`baselines/alphagen/requirements-arm64.txt`）：

| 包                                                               | 官方锁定 | 本机实际装的                      |
| ---------------------------------------------------------------- | -------- | --------------------------------- |
| numpy                                                            | 1.20.1   | 1.23.5                            |
| pandas                                                           | 1.2.4    | 1.5.3                             |
| matplotlib                                                       | 3.3.4    | 3.6.3                             |
| torch                                                            | 2.0.1    | 2.0.1（原样，arm64 有官方 wheel） |
| qlib / sb3_contrib / stable_baselines3 / gym / baostock / shimmy | 官方锁定 | 原样（均有 wheel，未改动）        |

**这份 `requirements-arm64.txt` 只用于本机 macOS 做代码/环境验证。在实验室服务器
（Linux x86_64 + CUDA）上，官方原始 `requirements.txt` 应该能直接装（那些包在
manylinux x86_64 上都有官方 wheel），没有必要用放宽版本，正式复现结果以服务器跑出来
的为准。**

## 在服务器上跑（正式复现，需要 mentor/你在远程执行）

本机没有 CSI300 数据也没有 GPU，数据已确认在实验室服务器上，跑通请按下面步骤：

1. **环境**：服务器建一个独立 conda/venv（Python 3.8 或 3.9），
   `cd baselines/alphagen && pip install -r requirements.txt`（原始锁定版本，Linux
   x86_64 上不需要放宽）。
2. **数据**（若还没有 qlib 格式的 CSI300 数据）：仓库 README 说明分两部分——
   - qlib 的**元数据**（不是行情本身）：按 [qlib 官方数据准备流程](https://github.com/microsoft/qlib#data-preparation) 走一遍。
   - 真实行情用 baostock（作者说 qlib 自带数据源时效性/准确性有顾虑）：跑
     `python data_collection/fetch_baostock_data.py`，默认存到
     `~/.qlib/qlib_data/cn_data_baostock_fwdadj`。路径可改，但要跟
     `scripts/rl.py` 里 `initialize_qlib(...)` 传的路径对上。
   - 如果服务器上已有别人跑过的 CSI300 qlib 数据，直接确认路径、跳过这步。
3. **跑训练**：
   - **GPU 空闲时**：`nvidia-smi` 看哪张卡空、记下卡号，用 `CUDA_VISIBLE_DEVICES` 锁卡
     （跟之前 RL 挖因子锁卡的做法一致），代码里的 `device` 不用碰。
   - **GPU 都被占用时（当前情况）**：`scripts/rl.py:194` 已经从 `torch.device("cuda:0")`
     改成 `torch.device("cpu")`——这个改动是有依据的，不是将就：策略网络是个很小的
     LSTM，真正耗时的是每步算因子 IC 那部分 pandas 计算，跟你们自己挖 RL 因子时的发现
     一致（"RL 瓶颈在 CPU、非 GPU"），GPU 在这里加速有限，纯 CPU 应该跑得动。等哪天
     GPU 又空出来了，把这行改回 `torch.device("cuda:0")` 配合 `CUDA_VISIBLE_DEVICES`
     锁卡即可，两种模式随时切换。

     代码里 `StockData`/`MseAlphaPool`/`AlphaEnv`/LSTM 特征提取器/`MaskablePPO` 构造时都
     显式传了 `device=device`，没有漏传、没有依赖类定义里那些 cuda:0 默认值，逻辑上确认
     不会碰 GPU。但不完全依赖"代码看得够仔细"，加一道硬保险——启动时带
     `CUDA_VISIBLE_DEVICES=""`，让进程在驱动层面直接看不到任何 GPU，physically 摸不到，
     不会跟别人的任务抢：

   ```bash
   cd baselines/alphagen
   CUDA_VISIBLE_DEVICES="" python -m scripts.rl --pool_capacity 10 --steps 5000   # 先冒烟：小规模，看看 CPU 上跑多快
   CUDA_VISIBLE_DEVICES="" python -m scripts.rl --pool_capacity 20                 # 正式：论文默认配置，steps 按容量自动选（20→250000）
   ```

   `pool_capacity`（因子池容量）、`instruments`（默认 `csi300`）、`steps`（不填按
   `{10:200000, 20:250000, 50:300000, 100:350000}` 自动选）是最常改的几个参数。
   **建议先跑冒烟档计个时**，纯 CPU 下 250000+ 步的正式档可能要跑较久，心里有个数
   再决定要不要缩小 `steps` 或等 GPU。

4. **产出**：`out/results/<instruments>_<pool_capacity>_<seed>_<timestamp>_rl/` 下每隔
   一段步数存一次 `{step}_steps_pool.json`（因子池表达式 + 权重，`{"exprs": [...],
"weights": [...]}`）和对应的模型 checkpoint、TensorBoard 日志。这个 json 就是下一步
   桥接需要的东西。

## 桥接回本项目（已就绪，等服务器产出后跑）

`src/rl/alphagen_bridge.py::parse_alphagen_expr` 把 AlphaGen 的表达式字符串（如
`Mean($close,5d)`）解析成本项目 `Node`。算子映射：

| AlphaGen                                                    | 本项目              | 备注                                                                                                                                                 |
| ----------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Add/Sub/Mul/Div                                             | add/sub/mul/div     | 一一对应                                                                                                                                             |
| Abs/Log                                                     | abs/log             | 一一对应                                                                                                                                             |
| CSRank                                                      | rank                | 截面排名，一一对应                                                                                                                                   |
| Ref(x,Nd)                                                   | delay(x,N)          |                                                                                                                                                      |
| Mean/Std(x,Nd)                                              | ts_mean/ts_std(x,N) |                                                                                                                                                      |
| Delta(x,Nd)                                                 | delta(x,N)          |                                                                                                                                                      |
| Max/Min(x,Nd)                                               | ts_max/ts_min(x,N)  | 顺手核实：`ts_max` 之前有个 copy-paste bug（调用了 `.rolling().min()`），本次复现顺手修了（`src/core/operators.py`），影响所有历史生成器，不只是这次 |
| Rank(x,Nd)（滚动分位数）                                    | ts_rank(x,N)        |                                                                                                                                                      |
| Corr(x,y,Nd)                                                | ts_corr(x,y,N)      |                                                                                                                                                      |
| Sign/Pow/Greater/Less/Sum/Var/Skew/Kurt/Med/Mad/WMA/EMA/Cov | —                   | 本项目 `operators.py` 没有对应算子，判不可翻译、跳过计数，不强行扩算子凑覆盖率                                                                       |
| `$vwap`                                                     | —                   | 本项目宽表没有 vwap 字段，同样跳过                                                                                                                   |
| 数值常量（如 `Add(x,5.0)`）                                 | —                   | 本项目算子表没有常量叶子节点，跳过                                                                                                                   |

拿到服务器产出的 `xxx_steps_pool.json` 后：

```bash
python scripts/bridge_alphagen.py path/to/xxx_steps_pool.json configs/default.yaml
```

会打印翻译覆盖率、跳过原因，落盘 `factor_library.pkl`（同构 schema，tag `alphagen`），
`n_explored` 记的是官方池子导出的候选总数（含翻译失败的，不为了让分母好看而剔除）。

本机用假数据（4 条表达式，2 条可翻译、2 条判不可翻译）跑通过这条链路，逻辑无误；
真实覆盖率、真实 IC 数字要等服务器产出真实 pool json 后才有意义。

## 已知风险/局限（如实记录，不是失败）

- CSI300（qlib/baostock）跟本项目的 hs300（tushare）不是同一套数据管线：官方数字和
  桥接后数字天然不完全等价，因此两份分开报告。
- 官方仓库代码写死 `cuda:0`，无 GPU 环境会直接报错——本机只验证了 import/环境，
  没有跑真实训练；真实训练与结果都在服务器上完成。
- 部分 AlphaGen 算子（尤其 Skew/Kurt/WMA/EMA/Cov 这类本项目没有的）会拉低桥接覆盖率，
  这是预期内的信息损失，覆盖率本身也是一个值得在论文里报告的数字。
- `run scripts as modules`（`python -m scripts.rl`）是仓库自己的要求，不是 `python scripts/rl.py`。
