# AlphaGen 复现

> 论文：_Generating Synergistic Formulaic Alpha Collections via Reinforcement
> Learning_（KDD 2023 ADS track）。官方仓库 [RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen)。

## 路线转折：放弃 qlib/baostock，改喂本项目自己的 tushare 数据

最初设计是"官方口径 + 桥接口径"两条线（跑官方 CSI300 数据拿论文口径数字，再把因子表达式
翻译回本项目数据算一份可比数字）。但服务器上 qlib/baostock 这条数据链路完全走不通——
连续踩了四个独立的坑（见下面"服务器踩坑记录"）：baostock 私有协议端口 10030 被计算节点
防火墙挡死、qlib 官方数据源（Azure blob）被禁、pyqlib 最新版 ABI 不兼容、脚本自己的路径
写错——最后一个网络层的坑（baostock 端口被挡）**无法通过改代码绕开，只能换节点**，但换
登录节点这条路径进一步排查后也不确定可行。

于是改路线：**AlphaGen 的算子求值只依赖一个鸭子类型接口，不必须是真的 qlib 数据**——
`alphagen/data/expression.py::Expression.evaluate(data, period)` 只用得到
`data.data`（`(days, features, stocks)` 张量）、`data.max_backtrack_days`、
`data.max_future_days`、`data.n_days`、`data.n_stocks` 这几个属性，不牵扯任何 qlib
内部实现。新增 `alphagen_generic/tushare_data.py::TushareStockData` 顶替官方的
`alphagen_qlib.stock_data.StockData`，直接从本项目的 `data/cache/prices_clean.parquet`
构造同接口对象；`alphagen_generic/tushare_calculator.py::TushareStockDataCalculator`
是 `QLibStockDataCalculator` 的鸭子类型子类（逻辑完全复用，只换个名字避免误导）。
`scripts/rl.py` 里 `initialize_qlib(...)` 调用整个删掉、`get_dataset()` 换成
`TushareStockData(...)`、训练/测试 `segments` 改成跟本项目 GP/RL/LLM/QuantFactor 同一套
`train_start=2021-01-01 / train_end=2023-01-01` 起始约定（也顺带避开了
[regime 断点](./regime断点.md) 提到的 2020→2021 断点）。

**代价与好处**：不再有"官方 CSI300 口径"这个单独的数字，全部就是 tushare/hs300 数据、
跟本项目其他四个生成器天然可比，`scripts/bridge_alphagen.py` 桥接脚本还是要跑——不是
因为数据不同要翻译，而是因为 AlphaGen 训练产出的 `{step}_steps_pool.json` 用的还是它
自己的表达式语法（`Mean($close,5d)` 这类），要转成本项目 `Node` 才能进
`compare_generators.py` 的统一 schema。

本机（macOS，无 GPU、无真实数据）已经用宽日期范围的合成数据把 `scripts.rl` 整条链路
（`TushareStockData` 构造 → `Expression.evaluate` → `TensorAlphaCalculator` 算 IC →
`AlphaEnv`/`MaskablePPO` 训练 → 因子池/checkpoint 落盘）跑通、退出码 0，pool 表达式、
IC 指标都正常输出。服务器上应该可以直接复用这条路径，完全不需要再碰 qlib/baostock。

## 服务器上跑（新路线，不需要 qlib/baostock）

1. **环境**：`cd baselines/alphagen && pip install -r requirements.txt`（已加
   `pyarrow`、`tensorboard` 两个新依赖，见下方"依赖变化"）。
2. **数据**：确认本项目自己的 `data/cache/prices_clean.parquet` 在服务器上是最新的
   （跟 GP/RL/LLM/QuantFactor 用的是同一份，不需要另外准备）。
3. **跑训练前先挑空卡/决定用不用 GPU**：
   - **GPU 空闲时**：`nvidia-smi` 看哪张卡空、记下卡号，`scripts/rl.py:197` 的
     `device = torch.device("cpu")` 改回 `torch.device("cuda:0")`，配合
     `CUDA_VISIBLE_DEVICES` 锁卡（跟之前 RL 挖因子锁卡的做法一致）。
   - **GPU 都被占用/不确定时（当前默认）**：保持 `torch.device("cpu")` 不动——策略网络是个
     很小的 LSTM，真正耗时的是每步算因子 IC 那部分张量计算，纯 CPU 应该跑得动，且启动时带
     `CUDA_VISIBLE_DEVICES=""` 在驱动层面直接不占用任何 GPU。

   ```bash
   cd baselines/alphagen
   CUDA_VISIBLE_DEVICES="" python -m scripts.rl --pool_capacity 10 --steps 5000   # 先冒烟
   CUDA_VISIBLE_DEVICES="" python -m scripts.rl --pool_capacity 20                 # 正式（论文默认配置）
   ```

   `pool_capacity`/`steps`（不填按 `{10:200000, 20:250000, 50:300000, 100:350000}`
   自动选）是最常改的参数；`instruments` 参数现在不生效（universe 已经在
   `prices_clean.parquet` 生成时定死是 hs300），不用传。
4. **产出**：`out/results/<...>_rl/` 下的 `{step}_steps_pool.json`（因子池表达式+权重）。
   拿到后跑桥接：

   ```bash
   python scripts/bridge_alphagen.py path/to/xxx_steps_pool.json configs/default.yaml
   ```

   落盘 `factor_library.pkl`（tag `alphagen`），可以直接进 `compare_generators.py` 五方对比
   （gp/rl/llm/quantfactor/alphagen）。算子映射表、覆盖率说明见下方"桥接回本项目"。

## 依赖变化（相对官方原始 requirements.txt）

| 包 | 官方原版 | 现在 | 原因 |
| --- | --- | --- | --- |
| numpy | 1.20.1 | 1.21.6 | gymnasium（shimmy 间接依赖）下限 1.21，官方锁定版本没考虑到传递依赖后来抬高下限 |
| qlib | `qlib==0.0.2.dev20`（假包，2018 年无关废弃小包） | `pyqlib==0.9.0` | PyPI 上叫 "qlib" 的不是微软的包；且不用最新 `pyqlib`（ABI 不兼容） |
| pyarrow | 无 | 新增 | `TushareStockData` 读 `.parquet` 要用 |
| tensorboard | 无（官方漏列） | 新增 | `sb3_contrib.MaskablePPO` 训练日志要用 |

## 桥接回本项目

`src/rl/alphagen_bridge.py::parse_alphagen_expr` 把 AlphaGen 的表达式字符串（如
`Mean($close,5d)`）解析成本项目 `Node`。算子映射：

| AlphaGen | 本项目 | 备注 |
| --- | --- | --- |
| Add/Sub/Mul/Div | add/sub/mul/div | 一一对应 |
| Abs/Log | abs/log | 一一对应 |
| CSRank | rank | 截面排名，一一对应 |
| Ref(x,Nd) | delay(x,N) | |
| Mean/Std(x,Nd) | ts_mean/ts_std(x,N) | |
| Delta(x,Nd) | delta(x,N) | |
| Max/Min(x,Nd) | ts_max/ts_min(x,N) | 顺手核实：`ts_max` 之前有个 copy-paste bug（调用了 `.rolling().min()`），本次复现顺手修了（`src/core/operators.py`），影响所有历史生成器，不只是这次 |
| Rank(x,Nd)（滚动分位数） | ts_rank(x,N) | |
| Corr(x,y,Nd) | ts_corr(x,y,N) | |
| Sign/Pow/Greater/Less/Sum/Var/Skew/Kurt/Med/Mad/WMA/EMA/Cov | — | 本项目 `operators.py` 没有对应算子，判不可翻译、跳过计数，不强行扩算子凑覆盖率 |
| `$vwap` | — | 本项目宽表没有 vwap 字段（`TushareStockData` 里是 `amount/volume` 近似算出来喂给 AlphaGen 自己用的，不进本项目 `Node` 体系），同样跳过 |
| 数值常量（如 `Add(x,5.0)`） | — | 本项目算子表没有常量叶子节点，跳过 |

`python scripts/bridge_alphagen.py path/to/xxx_steps_pool.json configs/default.yaml`
会打印翻译覆盖率、跳过原因，落盘 `factor_library.pkl`（同构 schema，tag `alphagen`），
`n_explored` 记的是官方池子导出的候选总数（含翻译失败的，不为了让分母好看而剔除）。
本机用假数据验证过这条链路逻辑无误；真实覆盖率、真实 IC 数字要等服务器产出真实
pool json 后才有意义。

## 服务器踩坑记录（如实记录，最终绕开而非解决）

这几个坑最后没有一个个"修好"，而是整体绕开（改用 tushare 数据），但记录下来避免以后
再摸黑重踩：

- **baostock 私有协议端口被挡**：baostock 不走 HTTP，用自己的私有 TCP 协议，服务器地址
  硬编码在库里（`baostock/common/contants.py`：`BAOSTOCK_SERVER_IP="www.baostock.com"`，
  `BAOSTOCK_SERVER_PORT=10030`）。计算节点 HTTPS(443)/ICMP 都通，但 10030 这种非常规端口
  大概率被防火墙挡了（错误码 `10002007` 对应 baostock 源码里的 `BSERR_RECVSOCK_FAIL`，
  socket 层面收不到数据，非代码 bug）——这是最终放弃 baostock 路线的直接原因。
- **qlib 官方数据源被禁**：Azure blob 存储当前公开访问返回 409（GitHub 上有多个 issue 在
  报告同样问题），`python -m qlib.cli.data qlib_data ...` 这条官方文档给的命令实测拉不下来。
- **`qlib` 是假包**：PyPI 上叫 "qlib" 的是 2018 年一个无关的废弃小包（"A Q Library for
  Data Scientist"），版本号刚好停在 `0.0.2.dev20`——跟 AlphaGen 官方仓库锁定的版本号一字
  不差，说明官方仓库这行从一开始就锁错了包名。微软真正的 qlib 发布包名是 `pyqlib`。
- **`pyqlib` 最新版 ABI 不兼容**：`pyqlib==0.9.7`（2025-08 发布）编译的 Cython 扩展
  （`_libs/rolling`）链接的是新版 numpy C-API，跟环境里锁的 `numpy==1.21.6` 不兼容，
  `import` 时报 `ImportError: numpy.core.multiarray failed to import`。改锁
  `pyqlib==0.9.0`（2022-12，跟 AlphaGen 仓库同时代）解决。
- **脚本自己的路径写错**：`scripts/rl.py` 里 `initialize_qlib("~/.qlib/qlib_data/cn_data")`
  跟 `data_collection/fetch_baostock_data.py` 实际 dump 数据的目录
  `~/.qlib/qlib_data/cn_data_2024h1` 对不上（`gp.py` 用的路径反而是对的，仓库自己内部不
  一致）。这个问题已随整体改路线一起失效（不再需要 qlib 数据）。
- `.gitignore` 里 `env/`（想忽略虚拟环境）没锚定根目录，把 `alphagen/rl/env/`（AlphaGen
  真正的强化学习环境代码）误伤漏提交，导致服务器 `git pull` 后 `ModuleNotFoundError:
  alphagen.rl.env`——已改成 `/env/` 锚定根目录并把文件补提交。
- `gym`（不是 `gymnasium`）每次 `import` 都会打印一段 numpy 2.0 不兼容的固定 deprecation
  警告，不是真的报错，可以忽略。
- `run scripts as modules`（`python -m scripts.rl`）是仓库自己的要求，不是
  `python scripts/rl.py`。
