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

# 配置数据源（首次使用）
cp .env.example .env   # 编辑填入 TUSHARE_TOKEN

# 下载行情数据（首次运行，约 10 分钟）
python scripts/fetch_data.py --universe hs300 --start 2018-01-01 --end 2024-12-31

# 运行 GP 因子挖掘
python scripts/run_gp.py configs/default.yaml
```

## 数据管理

### 数据源

通过项目根目录 `.env` 文件切换数据源：

| 数据源  | 说明                             | 配置                                             |
| ------- | -------------------------------- | ------------------------------------------------ |
| akshare | 免费无需注册，部分服务器网络受限 | `DATA_SOURCE=akshare`                            |
| tushare | 稳定，需注册获取 token           | `DATA_SOURCE=tushare` `TUSHARE_TOKEN=your_token` |

tushare token 在 [tushare.pro](https://tushare.pro) 注册后个人中心获取。

### 缓存结构

数据以 Parquet 格式缓存，**逐股存储，支持断点续传**，中途中断重新运行自动跳过已下载的：

```text
data/cache/
├── universe_hs300.txt               # 成分股列表（7天TTL自动刷新）
└── stocks/
    ├── akshare/
    │   └── 000001_2018-01-01_2024-12-31.parquet
    └── tushare/
        └── 000001_2018-01-01_2024-12-31.parquet
```

### 数据格式

`load_daily_prices()` 返回前复权日频数据：

```txt
MultiIndex: (date: Timestamp, code: str)
Columns:    open, high, low, close, volume, amount, turnover
```

### 限速说明

- akshare：0.5s/次（服务器 IP 可能被限，建议用 tushare）
- tushare 免费版：2s/次（上限 50次/分钟）

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
- [x] A 股数据接口对接（akshare / tushare，断点续传）
- [ ] 适应度函数实现
- [ ] 首次进化实验
