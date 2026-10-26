# EE6008：单股票下一交易日收益预测

本项目用个股和 Nasdaq 市场历史信息，预测下一交易日的收盘收益率，并辅助评价涨跌方向。目前已有 AAPL 和 TSLA 实验。

## 1. 工作内容

- 从原始 CSV 构造共同数据集，支持每天 8 或 6 个特征、最近 60 个交易日作为输入；默认以8特征为主。
- 实现 XGBoost、LightGBM、PatchTST、LSTM、GRU 单目标适配模型，以及零收益、训练期平均收益、延续前一日收益三种基线。
- 完成首轮及一轮超参数调优后的五模型验证；按日期保存预测、指标、配置、训练记录、预处理参数和模型权重。
- 核心代码使用 `.py`，Notebook 仅用于展示结果。
- `python main.py` 执行训练和验证；`python main.py test` 加载已保存的五个模型执行最终测试，不重新训练。另有 Notebook 比较指定的调优前后验证实验。

训练期为 2010—2021 年，验证期为 2022—2023 年；2024—2025 年进行最终测试

## 2. 配置环境

以下流程适用于 Windows / Linux x86-64，使用 conda 管理 Python、pip 安装项目包。先安装 [Miniforge](https://github.com/conda-forge/miniforge)，再打开能够使用 `conda` 的终端。已有可运行的 EE6008 环境可以直接跳到第 3 节。

### 2.1 打开项目并创建环境

把终端工作目录切换到项目根目录，也就是能看到 `main.py` 的目录。如果终端当前在项目的上一级，可运行：

```bash
cd stock_prediction
```

从空环境开始：

```bash
conda create -n EE6008 python=3.12 pip -y
conda activate EE6008
```

如果同名环境已经存在，使用现有环境，或者为新环境换一个名称，并在后续激活、内核选择时使用该名称。项目不依赖特定环境名称。

### 2.2 安装 PyTorch：下面两种方式选一种

**有兼容 NVIDIA GPU 和驱动，使用 GPU 训练：**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
```

**没有 NVIDIA GPU，或准备使用 CPU：**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

本项目已经在 Windows、RTX 5070 Ti Laptop GPU、PyTorch 2.14.0+cu132 上运行过。CPU 与 CUDA 13.2 安装包可在 [CPU 官方索引](https://download.pytorch.org/whl/cpu/torch/) 和 [CUDA 13.2 官方索引](https://download.pytorch.org/whl/cu132/torch/) 查询。其他硬件或操作系统请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择对应构建。


### 2.3 安装其余依赖

```bash
python -m pip install -r requirement.txt
python -m pip check
```

保留现有文件名 `requirement.txt`。其中固定了 PyTorch、NumPy、pandas、scikit-learn、matplotlib、XGBoost、LightGBM 和 Notebook 内核 ipykernel 的版本；先前安装的匹配 CPU / CUDA 构建满足 PyTorch 的版本要求。它是直接依赖清单，不是全部间接依赖的完整锁文件。

### 2.4 检查运行环境

```bash
python -c "import torch, numpy, pandas, sklearn, matplotlib, xgboost, lightgbm; print('torch:', torch.__version__); print('xgboost:', xgboost.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU 安装下 `CUDA: False` 是正常结果。准备使用 GPU 时，应显示 `CUDA: True`，并进一步确认能实际进行 GPU 运算：

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(32,32,device='cuda'); print((x @ x).mean().item()); torch.cuda.synchronize()"
```

默认 XGBoost、LightGBM 使用 CPU，PatchTST、LSTM 和 GRU 的 `device=auto` 在 CUDA 可用时使用 GPU，否则使用 CPU。仅运行树模型或基线时，启动入口不会导入 PyTorch。

## 3. 准备数据

数据随仓库共享，克隆或更新仓库后保留下面的相对位置：

```text
stock_prediction/
└── data/
    ├── AAPL_Apple_stock_data.csv
    └── TSLA_Tesla_stock_data.csv
```

默认配置读取 AAPL 原始 CSV；`configs/tsla.json` 读取 `data/TSLA_Tesla_stock_data.csv`。两者都包含个股和 Nasdaq 行情。`data/` 只保留这两份原始 CSV；未使用的 `with_features`、`standardized` 数据集和 `__MACOSX` 元数据目录已移除。特征由公共代码重新计算，标准化统计量仅在训练数据上拟合，无需另外准备预处理后的 CSV。

检查文件是否存在，并打印文件指纹：

```bash
python -c "from pathlib import Path; import hashlib; files=['AAPL_Apple_stock_data.csv','TSLA_Tesla_stock_data.csv']; [print(name, hashlib.sha256((Path('data') / name).read_bytes()).hexdigest()) for name in files]"
```

当前参考数据的 SHA-256 为：

```text
AAPL_Apple_stock_data.csv
5628d6886f2b3da80fd3ace8884ef90876c1a664a7eaa82ca90e5d952dff44cc
TSLA_Tesla_stock_data.csv
9d4198faa8a2c5d05ad8c754907062872d37f587f58d918c269dbf978874e545
```

实验中的 `data_sha256` 只对 `data_path` 指定的 CSV 文件内容计算，不对整个目录计算。删除未使用的其他文件不会改变此指纹；保留这两份原始文件的相对路径和内容，即可继续核对现有实验。

不同成员的指纹不一致时，先确认数据版本再比较结果。数据来源、Nasdaq 具体代码和价格复权口径仍待组员补充。

## 4. 完成本地实验：修改 JSON → 运行模型 → 查看结果

以下命令均在项目根目录（包含 `main.py`）执行，并先激活环境：

```bash
conda activate EE6008
```

### 4.1 运行项目检查

首次配置环境或修改代码后，可运行：

```bash
python -m unittest discover -s tests -v
```

检查集中在 `tests/test_models.py`，验证数据对齐、标准化、指标及模型接口等。

### 4.2 修改 JSON 配置

**先选择股票配置，再修改自己负责模型的参数：** AAPL 使用 `configs/aapl.json`，TSLA 使用 `configs/tsla.json`。每个文件包含公共数据设置和五个模型的参数段。

本轮小组调参统一基于 **8 特征**, 开始前请确保的字段为：

```json
"nasdaq_features": "all"
```


**公共配置项：**

| 字段 | 含义与修改方法 |
|---|---|
| `ticker` / `data_path` | 股票名称 / 原始 CSV 的项目相对路径；两者应对应。换股票优先用 `--config` 选择对应文件，不能只改股票名称 |
| `nasdaq_features` | `"all"` 为 8 特征，`"return_volume"` 为 6 特征，具体见下表 |
| `start` / `train_end` / `validation_end` | 数据起始日、训练结束日、验证结束日；按预测目标日期划分。本轮调参保持原设置 |
| `test_end` / `holdout_end` | 最终测试期及额外保留期的约定日期；普通训练命令不评价测试集，`main.py test` 使用源实验快照中的测试边界 |
| `window` | 历史输入交易日数，目前为 60；本轮调参保持一致 |
| `seed` | 随机种子，目前为 42；本轮比较保持一致，多种子实验另行约定 |

**切换 8 / 6 特征只需修改 `nasdaq_features`：**

| 配置值 | 个股特征 | Nasdaq 特征 | 总数 |
|---|---|---|---|
| `"all"` | `return`, `vol_change`, `intraday_return`, `range` | 同样四项，加 `nasdaq_` 前缀 | 8 |
| `"return_volume"` | 同上 | `nasdaq_return`, `nasdaq_vol_change` | 6 |

`nasdaq_vol_change = nasdaq_volume[t] / nasdaq_volume[t-1] - 1`，是成交量变化率。两种配置使用同一原始 CSV，无需手动删列；标准化仍由代码根据训练数据完成。

**模型参数项：**

| JSON 段落 | 常用可调参数及含义 |
|---|---|
| `xgboost` | `max_depth`：树深；`min_child_weight`：子节点最小权重约束；`n_estimators`：最大树数量；`learning_rate`：学习率；`subsample` / `colsample_bytree`：样本 / 特征抽样比例；`reg_lambda`：L2 正则化强度 |
| `lightgbm` | `num_leaves` / `max_depth`：树复杂度；`min_child_samples`：叶节点最少样本数；`n_estimators`、`learning_rate`、抽样比例、`reg_alpha` / `reg_lambda`：L1 / L2 正则化强度 |
| `patchtst` | `patch_len` / `stride`：时间片段长度 / 步长；`d_model`：表示维度；`n_heads`：注意力头数；`n_layers`：编码层数；`d_ff`：前馈网络宽度 |
| `lstm` / `gru` | `hidden_size`：隐藏状态维度；`num_layers`：循环网络层数 |
| 三种神经网络共用 | `learning_rate`：学习率；`dropout`：训练时随机置零部分激活的比例；`weight_decay`：权重衰减；`batch_size`：每批样本数；`epochs`：最大训练轮数；`patience`：验证表现连续不改善多少轮后停止 |

树模型的 `early_stopping_rounds` 同样控制验证早停。`n_jobs` 控制树模型 CPU 并行线程数；神经网络的 `device` 可选 `"auto"`、`"cpu"`、`"cuda"`，使用 CUDA 需要环境和硬件支持。当前树模型配置使用 CPU。

参数需满足实现约束：PatchTST 要求 `d_model` 能被 `n_heads` 整除，`0 < stride <= patch_len <= window`，且 `(window - patch_len)` 能被 `stride` 整除；GRU 若设 `num_layers=1`，需同时设 `dropout=0`。



### 4.3 运行实验与命令行参数

JSON 保存后即可运行，不必把 JSON 中的设置重复写到命令行。以 AAPL 的单模型调参为例，按单模型模型选择一条命令：

```bash
python main.py --config configs/aapl.json --models xgboost --run-name aapl_xgboost_tune_01
python main.py --config configs/aapl.json --models lightgbm --run-name aapl_lightgbm_tune_01
python main.py --config configs/aapl.json --models patchtst --run-name aapl_patchtst_tune_01
python main.py --config configs/aapl.json --models lstm --run-name aapl_lstm_tune_01
python main.py --config configs/aapl.json --models gru --run-name aapl_gru_tune_01
```

TSLA 只需选择对应配置，例如：

```bash
python main.py --config configs/tsla.json --models xgboost --run-name tsla_xgboost_tune_01
```

| 参数 | 作用 |
|---|---|
| `--config` | 指定 JSON 配置路径；省略时读取 `configs/aapl.json`。也可读取自己复制、编辑的完整配置文件 |
| `--models` | 选择模型，可填写一个或多个：`xgboost lightgbm patchtst lstm gru`；省略时运行全部五个。`baselines` 表示只运行简单基线；任何模型实验都会附带简单基线 |
| `--run-name` | 指定 `artifacts/` 下的新结果目录名；仅允许英文字母、数字、下划线和连字符。不能与已有目录重名；省略时自动生成时间戳名称 |
| `--nasdaq-features` | 可选，临时覆盖 JSON 的特征设置，取值 `all` 或 `return_volume`；覆盖后的值会写入本次结果快照，不修改原配置文件 |
| `--help` | 查看入口支持的参数 |

优先级为 **命令行 `--nasdaq-features` > JSON 中的 `nasdaq_features` > 未设置时默认 `all`**。例如 JSON 已设为 `"all"` 就不需要额外参数；需要临时强制使用 8 特征时可以运行：

```bash
python main.py --config configs/aapl.json --models xgboost --nasdaq-features all --run-name aapl_xgboost_tune_02
```


### 4.4 查看结果：先看这三项

更新代码后，Notebook 建议先重启内核，再运行全部单元格。

| 文件 | 先关注什么 |
|---|---|
| `metrics.csv` | 比较各模型 RMSE、MAE（越低越好）及方向准确率 |
| `validation_comparison.png` | 直观看误差和方向表现 |
| `predictions.csv` | 每个目标日的真实收益和各模型预测，用于分析及小组合并 |

`mae`、`rmse` 使用小数收益单位，例如误差 `0.01` 对应 1 个百分点。图中已乘以 100。方向由收益是否大于 0 得到，平盘归入“不涨”。回归误差小与方向准确率高是两个不同目标，应分别报告。

完整结果包还包含：

| 文件 | 作用 |
|---|---|
| `metrics.json` | 完整指标和混淆矩阵；矩阵行是真实类别、列是预测类别，顺序为 `[不涨, 上涨]` |
| `direction_reference.json` | 用训练期多数方向作预测时的验证准确率 |
| `data_audit.json` | 样本数量、日期边界及无效窗口记录 |
| `preprocessing.json` | 特征顺序、训练均值和标准差、目标缩放参数 |
| `config.json` | 本次运行的配置快照，应以它解释对应实验 |
| `manifest.json` | 是否完成、数据和代码指纹、环境版本、随机种子及评价分区 |
| `training.json` | 训练耗时、设备、最佳轮次和学习曲线 |
| `xgboost.ubj` / `lightgbm.txt` / `patchtst.pt` / `lstm.pt` / `gru.pt` | 本次训练模型的最佳权重，需配合预处理参数使用；只生成所选模型文件 |

`r2` 相对验证集真实均值计算，`r2_vs_zero = 1-SSE/sum(y²)` 相对零收益预测计算。

### 4.5 使用 Notebook 展示

`notebooks/results_display/` 存放单次实验和最终测试的结果展示；`notebooks/result_comparison/` 存放 Nasdaq 特征、超参数调优和最终测试调优前后的对比。Notebook 会从当前工作目录逐级向上定位项目根目录，因此在项目根目录或这两个子文件夹中运行均可；实验目录名仍只填写 `artifacts/` 下的名称，无需添加 `../`。

单次实验可用 `notebooks/results_display/01_AAPL_original_5models.ipynb` 或 `notebooks/results_display/01_TSLA_original_5models.ipynb` 展示。两者使用相同的通用读取规则。单模型调参请设置 `REQUIRE_ALL_MODELS = False`。

在开头**必须填写**实验目录名，例如：

```python
RUN_NAME = "aapl_xgboost_tune_01"
REQUIRE_ALL_MODELS = False
```

展示另一轮实验时，只需将 `RUN_NAME` 改为 `artifacts/` 下实际存在的目录名称（例如 `"tsla_xgboost_tune_01"`），再从头运行。

- 股票名称由 `artifacts/<RUN_NAME>/config.json` 的 `ticker` 获取；数据路径、日期范围、窗口、随机种子和模型参数也来自这份快照。
- `REQUIRE_ALL_MODELS = True` 要求所选实验包含五种模型；查看单模型或部分模型实验时设为 `False`。
- 读取 `predictions.csv`、`metrics.json`、`training.json`、`manifest.json`、`preprocessing.json` 和 `direction_reference.json`，核对原始数据指纹、日期、真实收益及指标。
- 依次展示实验目录校验与资源记录、按方向准确率降序排列的结果表格，以及模型预测曲线。
- `RECENT_DAYS` 只控制右侧局部曲线的展示日数（默认 80），左侧展示完整验证期，不影响历史输入窗口或全期指标。
- 选择 EE6008 内核后从头运行。如一直转圈，可先单独执行 `print("kernel ready")` 检查内核是否响应。
- 原始 CSV 必须仍位于快照指定的位置，且与训练时数据指纹一致。新增股票只要遵循现有公共数据格式和实验输出约定，就能复用任一本 Notebook；改变特征定义或预测任务仍需同步公共代码。

现有参考配置下，AAPL 为 2,960 条训练样本、501 条验证样本；TSLA 为 2,838 条训练样本、501 条验证样本。实际数量应以所选实验记录为准。

**比较超参数调优前后：** 打开 `notebooks/result_comparison/hyperparameter_tuning_comparison.ipynb`，选择 EE6008 内核并从头运行。默认同时比较 AAPL、TSLA 的五模型结果：

```python
RUN_PAIRS = [
    ("aapl_compare_all_01", "aapl_compare_all_02"),
    ("tsla_compare_all_01", "tsla_compare_all_02"),
]
```

每一对按“调优前、调优后”填写 `artifacts/` 下的目录名。股票名称来自各自 `config.json`；切换实验或增加股票只需修改 `RUN_PAIRS`，然后重启内核并运行全部单元格。

该 Notebook 只读取保存的配置和指标，展示共用参数改动、调优前后的结果表格（含方向准确率改善率）及变化幅度柱状图，不重复校验数据指纹或重算指标。所有比较均为验证集结果。

### 4.6 最终测试：直接使用之前实验中已保存的五个模型

在项目根目录、EE6008 环境中运行（已有完整结果时直接查看，不必重复执行；输出目录不能重名）：

```bash
python main.py test --source-run aapl_compare_all_02 --run-name aapl_final_test_02
python main.py test --source-run tsla_compare_all_02 --run-name tsla_final_test_02
```

每条命令会执行以下流程：

1. 从 `--source-run` 指定目录读取 `config.json`、`preprocessing.json`、`training.json`、`manifest.json`、`direction_reference.json`、验证预测及五份模型权重。不会读取 `configs/` 当前配置。
2. 核对原始 CSV 指纹，恢复最佳树前缀与神经网络权重，先重现已有验证预测。若日期、标签、输入形状或预测不匹配则停止，不进行测试。
3. 仅评价 `validation_end < target_date <= test_end`，目前为 2024—2025 年；每天用预测日之前已经观测到的最近 60 个交易日预测下一交易日，窗口可以包含较早的测试期行情，但模型与缩放参数始终固定。
4. 对五个模型及三个简单基线计算测试指标。历史平均收益及多数方向仍来自原训练期。

可选参数：`--device cpu` 强制神经网络在 CPU 推理，`--device cuda` 强制 GPU，默认 `auto`；树模型使用 CPU。`--batch-size 128` 只改变推理批大小，不修改原训练参数。没有 `--config`、`--models` 或超参数覆盖选项，始终恢复源实验的全部五个模型。
如只检查能否恢复模型，暂不执行测试：

```bash
python main.py test --source-run aapl_compare_all_02 --check-only
python main.py test --source-run tsla_compare_all_02 --check-only
```
当前调优后模型的测试输出位于 `artifacts/aapl_final_test_02/` 和 `artifacts/tsla_final_test_02/`；现有 `final_test_01` 来自调优前的 `compare_all_01`。模型来源以 `manifest.json` 的 `source_run` 为准。

| 文件 | 用途 |
|---|---|
| `metrics.csv` / `metrics.json` | 五个模型和三个基线的测试指标 |
| `predictions.csv` | 测试目标日、真实收益、方向和模型预测 |
| `test_comparison.png` | 明确标注 Test 的误差及方向对比图 |
| `manifest.json` | 应为 `status=complete`、`evaluation_split=test`、`retrained=false`、`test_evaluated=true`、`holdout_evaluated=false`；记录源实验及文件指纹 |
| `config.json` / `preprocessing.json` | 沿用源实验的配置与训练期缩放参数 |
| `source_manifest.json` / `validation_replay.json` | 原实验记录，以及恢复模型后对已有验证预测的一致性检查 |
| `data_audit.json` / `inference.json` | 实际测试日期、有效样本数、跳过样本及推理设备、耗时 |
| `direction_reference.json` | 原训练期多数方向在测试集上的准确率 |

测试入口不输出新的训练记录或模型权重。 最终测试使用下方专用的 Notebook。

**汇总两只股票的最终测试：** 调优前使用 `notebooks/results_display/01_final_test_results.ipynb`，调优后打开 `notebooks/results_display/02_final_test_results.ipynb`，开头显式指定两个目录。实验流程、指标解释与结果分析见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。

```python
RUN_NAMES = ["aapl_final_test_02", "tsla_final_test_02"]
```

选择 EE6008 内核，重启并运行全部单元格；切换股票或结果只需修改 `RUN_NAMES`。只接受两个不同股票的完整测试实验。
页面展示各股票按方向准确率降序排列的结果表格（含简单基线）和模型预测曲线。每只股票独立成图，左侧为完整测试期，右侧为最近 80 个交易日。

**比较调优前后的最终测试：** 打开 `notebooks/result_comparison/final_test_tuning_comparison.ipynb`，手动设置 `RUN_PAIRS`，例如 `[("aapl_final_test_01", "aapl_final_test_02"), ("tsla_final_test_01", "tsla_final_test_02")]`。每一对依次为调优前、调优后的测试目录；只展示结果表格和变化幅度柱状图。表格中的方向准确率改善率为 `(调优后 − 调优前) / 调优前 × 100%`，柱状图中的方向准确率变化为百分点。




## 5. 文件结构

```text
stock_prediction/
├── main.py                   # 日常使用的实验入口
├── README.md                 # 当前进展和操作手册
├── PROJECT_PLAN.md           # 共同约定、成员任务、待同步事项
├── cooperation.md            # GitHub 分支与 PR 协作步骤
├── requirement.txt           # Python 直接依赖版本
├── configs/                 # aapl.json、tsla.json：股票数据及五模型参数
├── data/                     # 两份原始 CSV：AAPL、TSLA 的 stock_data
├── src/
│   ├── data.py               # 共同的数据处理
│   ├── baselines.py          # 简单预测基线
│   ├── evaluate.py           # 共同的指标和绘图
│   ├── inference.py          # 恢复五个已保存模型，仅推理
│   └── models/               # XGBoost、LightGBM、PatchTST、LSTM、GRU
├── scripts/run_experiment.py # 训练与验证入口
├── scripts/evaluate_test.py  # 冻结模型的最终测试入口
├── notebooks/
│   ├── results_display/     # 单次验证实验、双股票最终测试展示（8 个）
│   └── result_comparison/   # 特征、验证调参、最终测试调参对比（3 个）
├── tests/test_models.py      # 统一的数据、指标及模型检查
├── artifacts/                # 各次实验输出
└── private/                  # 个人笔记，不提交 Git
```



