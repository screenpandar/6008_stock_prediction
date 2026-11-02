# EE6008: Single-Stock Next-Trading-Day Return Prediction

[中文版 README](README_CN.md)

This project uses historical information about individual stocks and the Nasdaq market to predict the next trading day's closing return, with supplementary evaluation of price direction. Experiments are currently available for AAPL and TSLA.

## 1. Project Work

- Build a shared dataset from raw CSV files, using the most recent 60 trading days with either 8 or 6 features per day; 8 features are the default.
- Implement single-target model adapters for XGBoost, LightGBM, PatchTST, LSTM and GRU, together with three baselines: zero return, training-period mean return and previous-day return.
- Complete five-model validation experiments with the initial settings and after one round of hyperparameter tuning; save date-aligned predictions, metrics, configurations, training records, preprocessing parameters and model weights.
- Keep the core implementation in `.py` files and use notebooks only to display results.
- Use `python main.py` for training and validation; use `python main.py test` to load the five saved models for final testing without retraining. A separate notebook compares explicitly selected validation experiments before and after tuning.

The training period is 2010–2021, the validation period is 2022–2023, and the final test period is 2024–2025.

## 2. Environment Setup

The following steps apply to Windows / Linux x86-64, using conda to manage Python and pip to install project packages. Install [Miniforge](https://github.com/conda-forge/miniforge) first, then open a terminal where `conda` is available. If you already have a working EE6008 environment, skip to Section 3.

### 2.1 Open the Project and Create an Environment

Set the terminal's working directory to the project root, which contains `main.py`. If the terminal is currently in the project's parent directory, run:

```bash
cd stock_prediction
```

Start with a new environment:

```bash
conda create -n EE6008 python=3.12 pip -y
conda activate EE6008
```

If an environment with this name already exists, use it or choose another name for the new environment, then use that name when activating it and selecting the notebook kernel. The project does not require a particular environment name.

### 2.2 Install PyTorch: Choose One of the Following Options

**For GPU training with a compatible NVIDIA GPU and driver:**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
```

**If you do not have an NVIDIA GPU or intend to use the CPU:**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

This project has been run on Windows with an RTX 5070 Ti Laptop GPU and PyTorch 2.14.0+cu132. CPU and CUDA 13.2 packages can be found in the [official CPU index](https://download.pytorch.org/whl/cpu/torch/) and [official CUDA 13.2 index](https://download.pytorch.org/whl/cu132/torch/). For other hardware or operating systems, select the appropriate build using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).


### 2.3 Install the Remaining Dependencies

```bash
python -m pip install -r requirement.txt
python -m pip check
```

Keep the existing filename `requirement.txt`. It pins versions of PyTorch, NumPy, pandas, scikit-learn, matplotlib, XGBoost, LightGBM and the notebook kernel package ipykernel. The matching CPU / CUDA build installed above satisfies the PyTorch version requirement. This is a list of direct dependencies, not a complete lockfile for all transitive dependencies.

### 2.4 Check the Runtime Environment

```bash
python -c "import torch, numpy, pandas, sklearn, matplotlib, xgboost, lightgbm; print('torch:', torch.__version__); print('xgboost:', xgboost.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA: False` is expected for a CPU installation. If you intend to use the GPU, the output should show `CUDA: True`. Also confirm that GPU computation actually works:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(32,32,device='cuda'); print((x @ x).mean().item()); torch.cuda.synchronize()"
```

By default, XGBoost and LightGBM use the CPU. For PatchTST, LSTM and GRU, `device=auto` selects the GPU when CUDA is available and the CPU otherwise. When only tree models or baselines are selected, the experiment entry point does not import PyTorch.

## 3. Prepare the Data

The data is shared through the repository. After cloning or updating it, preserve the following relative paths:

```text
stock_prediction/
└── data/
    ├── AAPL_Apple_stock_data.csv
    └── TSLA_Tesla_stock_data.csv
```

The default configuration reads the raw AAPL CSV; `configs/tsla.json` reads `data/TSLA_Tesla_stock_data.csv`. Both files contain individual-stock and Nasdaq market data. Only these two raw CSV files remain in `data/`; the unused `with_features` and `standardized` datasets and the `__MACOSX` metadata directory have been removed. The shared code recalculates features and fits standardization statistics only on training data, so no separately preprocessed CSV files are required.

Check that the files exist and print their fingerprints:

```bash
python -c "from pathlib import Path; import hashlib; files=['AAPL_Apple_stock_data.csv','TSLA_Tesla_stock_data.csv']; [print(name, hashlib.sha256((Path('data') / name).read_bytes()).hexdigest()) for name in files]"
```

The SHA-256 fingerprints of the current reference data are:

```text
AAPL_Apple_stock_data.csv
5628d6886f2b3da80fd3ace8884ef90876c1a664a7eaa82ca90e5d952dff44cc
TSLA_Tesla_stock_data.csv
9d4198faa8a2c5d05ad8c754907062872d37f587f58d918c269dbf978874e545
```

An experiment's `data_sha256` is calculated from the contents of the CSV specified by `data_path`, not from the entire directory. Deleting other unused files does not change this fingerprint. Preserve the relative paths and contents of these two raw files to continue checking existing experiments.

If fingerprints differ between team members, confirm the data version before comparing results. The data source, exact Nasdaq symbol and price-adjustment conventions still need to be provided by the team.

## 4. Run a Local Experiment: Edit JSON → Run Models → View Results

Run all commands below from the project root, which contains `main.py`, and activate the environment first:

```bash
conda activate EE6008
```

### 4.1 Run the Project Checks

After setting up the environment for the first time or modifying the code, you can run:

```bash
python -m unittest discover -s tests -v
```

The checks are consolidated in `tests/test_models.py` and cover data alignment, standardization, metrics and model interfaces, among other items.

### 4.2 Edit the JSON Configuration

**Select the stock configuration first, then adjust the parameters of the model you are responsible for:** use `configs/aapl.json` for AAPL and `configs/tsla.json` for TSLA. Each file contains shared data settings and parameter sections for all five models.

This round of team tuning uses **8 features** consistently. Before starting, ensure the following setting is in place:

```json
"nasdaq_features": "all"
```


**Shared configuration fields:**

| Field | Meaning and How to Change It |
|---|---|
| `ticker` / `data_path` | Stock symbol / project-relative path to the raw CSV; the two should correspond. To switch stocks, use `--config` to select the appropriate file rather than changing only the stock symbol |
| `nasdaq_features` | `"all"` selects 8 features; `"return_volume"` selects 6 features. See the table below |
| `start` / `train_end` / `validation_end` | Data start date, training end date and validation end date; splits are based on prediction target dates. Keep the existing settings for this tuning round |
| `test_end` / `holdout_end` | Agreed end dates for the final test period and additional holdout period; ordinary training commands do not evaluate the test set, and `main.py test` uses the test boundaries in the source experiment's configuration snapshot |
| `window` | Number of historical trading days used as input, currently 60; keep it consistent during this tuning round |
| `seed` | Random seed, currently 42; keep it consistent for this comparison, with multi-seed experiments to be agreed separately |

**To switch between 8 and 6 features, change only `nasdaq_features`:**

| Configuration Value | Stock Features | Nasdaq Features | Total |
|---|---|---|---|
| `"all"` | `return`, `vol_change`, `intraday_return`, `range` | The same four features with the `nasdaq_` prefix | 8 |
| `"return_volume"` | Same as above | `nasdaq_return`, `nasdaq_vol_change` | 6 |

`nasdaq_vol_change = nasdaq_volume[t] / nasdaq_volume[t-1] - 1` is the rate of change in trading volume. Both configurations use the same raw CSV, so there is no need to delete columns manually. Standardization is still performed in code using training data.

**Model parameters:**

| JSON Section | Common Tunable Parameters and Their Meanings |
|---|---|
| `xgboost` | `max_depth`: tree depth; `min_child_weight`: minimum child-node weight constraint; `n_estimators`: maximum number of trees; `learning_rate`: learning rate; `subsample` / `colsample_bytree`: sample / feature sampling fractions; `reg_lambda`: L2 regularization strength |
| `lightgbm` | `num_leaves` / `max_depth`: tree complexity; `min_child_samples`: minimum number of samples in a leaf; `n_estimators`, `learning_rate`, sampling fractions, and `reg_alpha` / `reg_lambda`: L1 / L2 regularization strengths |
| `patchtst` | `patch_len` / `stride`: temporal patch length / step size; `d_model`: representation dimension; `n_heads`: number of attention heads; `n_layers`: number of encoder layers; `d_ff`: feed-forward network width |
| `lstm` / `gru` | `hidden_size`: hidden-state dimension; `num_layers`: number of recurrent layers |
| Shared by the three neural networks | `learning_rate`: learning rate; `dropout`: fraction of activations randomly set to zero during training; `weight_decay`: weight decay; `batch_size`: samples per batch; `epochs`: maximum number of training epochs; `patience`: number of consecutive epochs without validation improvement before stopping |

For tree models, `early_stopping_rounds` also controls validation-based early stopping. `n_jobs` controls CPU parallelism for tree models. Neural-network `device` options are `"auto"`, `"cpu"` and `"cuda"`; CUDA requires compatible hardware and an appropriate environment. The current tree-model configurations use the CPU.

Parameters must satisfy the implementation constraints: PatchTST requires `d_model` to be divisible by `n_heads`, `0 < stride <= patch_len <= window`, and `(window - patch_len)` to be divisible by `stride`. If GRU uses `num_layers=1`, set `dropout=0` as well.



### 4.3 Run Experiments and Use Command-Line Arguments

After saving the JSON file, you can run the experiment without repeating its settings on the command line. For single-model tuning on AAPL, select one command for the model you want to run:

```bash
python main.py --config configs/aapl.json --models xgboost --run-name aapl_xgboost_tune_01
python main.py --config configs/aapl.json --models lightgbm --run-name aapl_lightgbm_tune_01
python main.py --config configs/aapl.json --models patchtst --run-name aapl_patchtst_tune_01
python main.py --config configs/aapl.json --models lstm --run-name aapl_lstm_tune_01
python main.py --config configs/aapl.json --models gru --run-name aapl_gru_tune_01
```

For TSLA, select its configuration instead, for example:

```bash
python main.py --config configs/tsla.json --models xgboost --run-name tsla_xgboost_tune_01
```

| Argument | Purpose |
|---|---|
| `--config` | Path to the JSON configuration; defaults to `configs/aapl.json` when omitted. You can also supply a complete configuration file that you have copied and edited |
| `--models` | Select one or more models from `xgboost lightgbm patchtst lstm gru`; runs all five when omitted. `baselines` runs only the simple baselines; every model experiment also includes the simple baselines |
| `--run-name` | Name of a new result directory under `artifacts/`; only English letters, digits, underscores and hyphens are allowed. It must not duplicate an existing directory name; a timestamp-based name is generated when omitted |
| `--nasdaq-features` | Optional temporary override of the JSON feature setting, either `all` or `return_volume`; the overridden value is saved in the result snapshot without modifying the original configuration file |
| `--help` | Display the arguments supported by the entry point |

The precedence is **command-line `--nasdaq-features` > JSON `nasdaq_features` > default `all` when neither is set**. For example, if the JSON already specifies `"all"`, no extra argument is required. To temporarily force 8 features, run:

```bash
python main.py --config configs/aapl.json --models xgboost --nasdaq-features all --run-name aapl_xgboost_tune_02
```


### 4.4 View Results: Start with These Three Files

After updating the code, restart the notebook kernel before running all cells.

| File | What to Check First |
|---|---|
| `metrics.csv` | Compare model RMSE and MAE (lower is better), together with directional accuracy |
| `validation_comparison.png` | Visually compare return errors and directional performance |
| `predictions.csv` | Actual returns and model predictions for each target date, for analysis and integration of team results |

`mae` and `rmse` use decimal return units: an error of `0.01` corresponds to 1 percentage point. Values in the plots are multiplied by 100. Direction is determined by whether the return is greater than 0, with flat returns classified as “not up.” Low regression error and high directional accuracy are different objectives and should be reported separately.

The complete result package also contains:

| File | Purpose |
|---|---|
| `metrics.json` | Complete metrics and confusion matrices; rows represent actual classes and columns represent predicted classes, ordered as `[not up, up]` |
| `direction_reference.json` | Validation accuracy when predicting the majority direction from the training period |
| `data_audit.json` | Sample counts, date boundaries and invalid-window records |
| `preprocessing.json` | Feature order, training means and standard deviations, and target-scaling parameters |
| `config.json` | Configuration snapshot for this run; use it to interpret the corresponding experiment |
| `manifest.json` | Completion status, data and code fingerprints, environment versions, random seed and evaluation split |
| `training.json` | Training time, device, best iteration or epoch, and learning curves |
| `xgboost.ubj` / `lightgbm.txt` / `patchtst.pt` / `lstm.pt` / `gru.pt` | Best saved model weights from this training run, to be used with the preprocessing parameters; files are generated only for the selected models |

`r2` is calculated relative to the actual mean return in the validation set; `r2_vs_zero = 1-SSE/sum(y²)` is calculated relative to zero-return predictions.

### 4.5 Display Results in Notebooks

`notebooks/results_display/` contains displays for individual experiments and final tests. `notebooks/result_comparison/` contains Nasdaq feature comparisons, hyperparameter-tuning comparisons, and before/after-tuning comparisons on the final test set. Notebooks search upward from the current working directory to locate the project root, so they can run from either the project root or these two subdirectories. Specify only the experiment directory name under `artifacts/`; do not add `../`.

Use `notebooks/results_display/01_AAPL_original_5models.ipynb` or `notebooks/results_display/01_TSLA_original_5models.ipynb` to display an individual experiment. Both follow the same general loading rules. For single-model tuning, set `REQUIRE_ALL_MODELS = False`.

You **must specify** the experiment directory name near the beginning, for example:

```python
RUN_NAME = "aapl_xgboost_tune_01"
REQUIRE_ALL_MODELS = False
```

To display another experiment, change `RUN_NAME` to an existing directory name under `artifacts/`, such as `"tsla_xgboost_tune_01"`, then rerun the notebook from the beginning.

- The stock symbol is read from `ticker` in `artifacts/<RUN_NAME>/config.json`; the data path, date range, window, random seed and model parameters also come from this snapshot.
- `REQUIRE_ALL_MODELS = True` requires all five models in the selected experiment; set it to `False` to view an experiment containing one model or a subset of models.
- The notebook reads `predictions.csv`, `metrics.json`, `training.json`, `manifest.json`, `preprocessing.json` and `direction_reference.json`, checking the raw-data fingerprint, dates, actual returns and metrics.
- It displays experiment-directory checks and resource records, a result table sorted by directional accuracy in descending order, and model prediction curves.
- `RECENT_DAYS` controls only the number of days shown in the right-hand recent-period plots (80 by default). The left-hand plots show the full validation period. This setting does not affect the historical input window or full-period metrics.
- Select the EE6008 kernel and run the notebook from the beginning. If execution keeps spinning, first run `print("kernel ready")` in a separate cell to check whether the kernel responds.
- The raw CSV must remain at the location specified in the snapshot and match the data fingerprint recorded during training. Any new stock can use these notebooks if it follows the shared data format and experiment-output conventions. Changes to feature definitions or the prediction task must also be reflected in the shared code.

Under the current reference configuration, AAPL has 2,960 training samples and 501 validation samples; TSLA has 2,838 training samples and 501 validation samples. Use the selected experiment's records as the authority for actual counts.

**Compare results before and after hyperparameter tuning:** open `notebooks/result_comparison/hyperparameter_tuning_comparison.ipynb`, select the EE6008 kernel and run from the beginning. By default, it compares the five-model results for both AAPL and TSLA:

```python
RUN_PAIRS = [
    ("aapl_compare_all_01", "aapl_compare_all_02"),
    ("tsla_compare_all_01", "tsla_compare_all_02"),
]
```

Each pair contains directory names under `artifacts/` in the order “before tuning, after tuning.” Stock symbols come from the respective `config.json` files. To switch experiments or add stocks, modify `RUN_PAIRS`, then restart the kernel and run all cells.

This notebook reads only saved configurations and metrics. It displays shared parameter changes, before/after result tables (including relative improvement in directional accuracy), and bar charts showing the changes. It does not repeat data-fingerprint checks or recalculate metrics. All comparisons use validation results.

### 4.6 Final Testing: Use the Five Models Saved in Previous Experiments

Run the following from the project root in the EE6008 environment. If complete results already exist, view them directly rather than rerunning the tests; output directory names must be unique:

```bash
python main.py test --source-run aapl_compare_all_02 --run-name aapl_final_test_02
python main.py test --source-run tsla_compare_all_02 --run-name tsla_final_test_02
```

Each command performs the following steps:

1. Read `config.json`, `preprocessing.json`, `training.json`, `manifest.json`, `direction_reference.json`, validation predictions and the five sets of model weights from the directory specified by `--source-run`. The current configuration in `configs/` is not read.
2. Check the raw CSV fingerprint, restore the selected tree ensemble prefix and neural-network weights, and first reproduce the saved validation predictions. If dates, labels, input shapes or predictions do not match, stop without performing the test evaluation.
3. Evaluate only targets satisfying `validation_end < target_date <= test_end`, currently covering 2024–2025. Each next-trading-day prediction uses the most recent 60 trading days observed before the target date. The window may contain earlier test-period observations, but the model and scaling parameters remain fixed.
4. Compute test metrics for the five models and three simple baselines. The historical mean return and majority direction still come from the original training period.

Optional arguments: `--device cpu` forces neural-network inference on the CPU; `--device cuda` forces the GPU; the default is `auto`. Tree models use the CPU. `--batch-size 128` changes only the inference batch size, not the original training parameters. There are no `--config`, `--models` or hyperparameter-override options; all five models from the source experiment are always restored.
To check that the models can be restored without evaluating the test set:

```bash
python main.py test --source-run aapl_compare_all_02 --check-only
python main.py test --source-run tsla_compare_all_02 --check-only
```
The current tuned-model test outputs are in `artifacts/aapl_final_test_02/` and `artifacts/tsla_final_test_02/`. The existing `final_test_01` results come from the original, untuned `compare_all_01` experiments. Use `source_run` in `manifest.json` to identify the model source.

| File | Purpose |
|---|---|
| `metrics.csv` / `metrics.json` | Test metrics for the five models and three baselines |
| `predictions.csv` | Test target dates, actual returns, directions and model predictions |
| `test_comparison.png` | Error and direction comparison plots explicitly labeled Test |
| `manifest.json` | Should record `status=complete`, `evaluation_split=test`, `retrained=false`, `test_evaluated=true` and `holdout_evaluated=false`; also records the source experiment and file fingerprints |
| `config.json` / `preprocessing.json` | Configuration and training-period scaling parameters carried over from the source experiment |
| `source_manifest.json` / `validation_replay.json` | Original experiment record and consistency checks against the saved validation predictions after restoring the models |
| `data_audit.json` / `inference.json` | Actual test dates, valid sample counts, skipped samples, inference device and elapsed time |
| `direction_reference.json` | Test accuracy of the majority direction learned from the original training period |

The test entry point does not produce new training records or model weights. Use the dedicated notebooks below for final-test results.

**Display final-test results for both stocks:** use `notebooks/results_display/01_final_test_results.ipynb` for the original models and `notebooks/results_display/02_final_test_results.ipynb` for the tuned models, explicitly specifying the two directories near the beginning. See [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) for the experiment workflow, metric explanations and result analysis.

```python
RUN_NAMES = ["aapl_final_test_02", "tsla_final_test_02"]
```

Select the EE6008 kernel, restart it and run all cells. To switch stocks or results, change only `RUN_NAMES`. The notebook accepts complete test experiments for two different stocks only.
It displays a result table for each stock, including the simple baselines and sorted by directional accuracy in descending order, together with model prediction curves. Each stock has a separate figure: the full test period is on the left and the most recent 80 trading days are on the right.

**Compare final-test results before and after tuning:** open `notebooks/result_comparison/final_test_tuning_comparison.ipynb` and set `RUN_PAIRS` manually, for example `[("aapl_final_test_01", "aapl_final_test_02"), ("tsla_final_test_01", "tsla_final_test_02")]`. Each pair lists the test directories before and after tuning, in that order. Only result tables and bar charts showing the changes are displayed. Relative improvement in directional accuracy in the tables is calculated as `(after tuning − before tuning) / before tuning × 100%`; directional accuracy changes in the bar charts are expressed in percentage points.




## 5. File Structure

```text
stock_prediction/
├── main.py                   # Main entry point for everyday experiments
├── README.md                 # Current progress and operating instructions
├── PROJECT_PLAN.md           # Shared conventions, team tasks and items to coordinate
├── cooperation.md            # GitHub branch and PR collaboration workflow
├── requirement.txt           # Direct Python dependency versions
├── configs/                 # aapl.json and tsla.json: stock data and five-model parameters
├── data/                     # Two raw CSV files: AAPL and TSLA stock_data
├── src/
│   ├── data.py               # Shared data processing
│   ├── baselines.py          # Simple prediction baselines
│   ├── evaluate.py           # Shared metrics and plotting
│   ├── inference.py          # Restore the five saved models for inference only
│   └── models/               # XGBoost, LightGBM, PatchTST, LSTM and GRU
├── scripts/run_experiment.py # Training and validation entry point
├── scripts/evaluate_test.py  # Final-test entry point using frozen models
├── notebooks/
│   ├── results_display/     # Individual validation experiments and two-stock final-test displays (8)
│   └── result_comparison/   # Feature, validation tuning and final-test tuning comparisons (3)
├── tests/test_models.py      # Shared data, metric and model checks
├── artifacts/                # Experiment outputs
└── private/                  # Personal notes, excluded from Git
```
