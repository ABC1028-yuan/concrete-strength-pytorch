# Concrete Compressive Strength Prediction

基于 UCI Concrete Compressive Strength 数据集的回归实验项目。项目使用混凝土配合比和龄期等 8 个特征预测抗压强度（MPa），包含线性回归基线、PyTorch 多层感知机（MLP）、多随机种子消融、CPU/GPU 速度比较和独立预测入口。

## 数据集

- 来源：[UCI Machine Learning Repository — Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- 规模：1,030 条样本；8 个连续输入特征和 1 个连续目标；原始数据无缺失值。
- 输入列：`Cement`、`Blast Furnace Slag`、`Fly Ash`、`Water`、`Superplasticizer`、`Coarse Aggregate`、`Fine Aggregate`、`Age`。
- 目标列：`Concrete compressive strength`，单位 MPa。
- 原始来源文件位于 `data/raw/`；统一字段名后的数据位于 `data/concrete.csv`；前 5 条样本位于 `data/sample_preview.csv`。
- 数据引用：I.-C. Yeh, “Modeling of strength of high-performance concrete using artificial neural networks,” *Cement and Concrete Research*, 1998. DOI: [10.24432/C5PK67](https://doi.org/10.24432/C5PK67)。数据集许可为 CC BY 4.0。

## 项目结构

```text
concrete_strength/
├── data/                 # 整理后的数据、预测模板和 UCI 原始文件
├── figures/              # 训练曲线、预测散点图、消融图、测速图
├── results/              # 指标 CSV、运行记录、预测结果和模型权重
├── src/
│   ├── run_experiment.py       # 线性回归与基础 MLP 实验
│   ├── predict.py              # 加载模型进行 CSV 预测
│   ├── run_multiseed_ablation.py # 多种子消融实验
│   └── benchmark_cpu_gpu.py    # CPU/CUDA 训练速度比较
├── requirements.txt
├── 实验与报告方案.md
└── 第二次学习记录.md
```

## 环境准备

建议在项目目录使用独立虚拟环境。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 PyCharm 中也可以将项目解释器设置为项目的 `.venv`。CPU 运行不需要 CUDA；GPU 测速要求已安装与本机 CUDA 驱动匹配的 PyTorch CUDA 版本，并且 `torch.cuda.is_available()` 返回 `True`。请在运行测速脚本前确认设备可用。

## 运行实验

以下命令均从项目根目录运行。

### 基础模型和线性回归基线

```powershell
python src/run_experiment.py
```

程序读取 `data/raw/Concrete_Data.xls`，按固定种子 42 划分训练集、验证集和测试集（60%/20%/20%，分别为 618/206/206 条）。标准化器仅在训练集拟合。线性回归和 MLP 使用相同划分；MLP 为 `8 → 32 → 16 → 1`，采用 Adam、学习率 0.001、batch size 32、MSE 损失，最多训练 300 轮，并按验证集 RMSE 保存最佳权重。

输出包括 `results/metrics.csv`、训练历史、测试集预测、`results/mlp_best.pt` 以及 `figures/training_curve.png` 和 `figures/prediction_scatter.png`。

### 多随机种子消融实验

```powershell
python src/run_multiseed_ablation.py
```

固定数据划分种子 42，以训练种子 42、43、44 运行五种配置（共 15 次训练）：基础网络、关闭标准化、单隐藏层 `8→32→1`、较窄网络 `8→16→8→1`、学习率 0.01。每轮依据验证集 RMSE 选择权重，并在相同测试集计算 MAE、RMSE 和 R²。逐次结果、均值/标准差汇总、种子 42 验证曲线和测试 RMSE 对比图分别写入 `results/ablation_runs.csv`、`results/ablation_summary.csv`、`results/ablation_validation_curves_seed42.csv`、`figures/ablation_val_curves_seed42.png` 和 `figures/ablation_rmse.png`。

### CPU 与 GPU 速度比较

```powershell
python src/benchmark_cpu_gpu.py
```

脚本默认在每种设备上预热 5 轮，再分别训练 300 轮、重复 3 次。两边使用相同数据划分、基础 MLP、batch size 和学习率；计时不含数据传输，CUDA 计时前后会同步设备。结果写入 `results/cpu_gpu_benchmark_runs.csv`、`results/cpu_gpu_benchmark_summary.csv` 和 `figures/cpu_gpu_training_time.png`。

## 当前实验结果

### 基线模型测试集表现

| 模型 | MAE (MPa) | RMSE (MPa) | R² |
|---|---:|---:|---:|
| 线性回归 | 8.488 | 11.002 | 0.585 |
| MLP（8-32-16-1） | 5.309 | 6.893 | 0.837 |

该表来自固定划分种子 42、训练种子 42 的一次基础对照。MAE/RMSE 越低越好，R² 越接近 1 通常表示拟合越好。

### 三种子消融汇总

下表格式为均值 ± 标准差；验证 RMSE 是各种子所选最佳轮次的验证误差，测试指标统计 3 个训练种子的固定测试集结果。

| 配置 | 验证 RMSE (MPa) | 测试 MAE (MPa) | 测试 RMSE (MPa) | 测试 R² |
|---|---:|---:|---:|---:|
| 基础网络 8-32-16-1 | 6.003 ± 0.334 | 5.260 ± 0.136 | 6.870 ± 0.054 | 0.838 ± 0.003 |
| 不使用标准化 | 6.061 ± 0.217 | 5.358 ± 0.199 | 7.103 ± 0.205 | 0.827 ± 0.010 |
| 单隐藏层 8-32-1 | 8.569 ± 1.747 | 7.113 ± 1.279 | 9.121 ± 1.516 | 0.709 ± 0.098 |
| 较窄网络 8-16-8-1 | 8.414 ± 1.872 | 7.010 ± 1.266 | 9.115 ± 1.453 | 0.710 ± 0.093 |
| 学习率 0.01 | 5.151 ± 0.250 | 4.680 ± 0.472 | 6.398 ± 0.452 | 0.859 ± 0.020 |

在本次配置和固定划分下，学习率 0.01 的配置取得最低平均测试 RMSE；两种缩小网络的平均误差较高且种子间波动更大。每种配置仅运行 3 个种子，结果用于课程实验比较，不代表对其他划分或数据的普遍结论。验证 RMSE 曲线见 `figures/ablation_val_curves_seed42.png`，测试 RMSE 均值和标准差图见 `figures/ablation_rmse.png`。

### CPU/GPU 测速

| 设备 | 设备型号 | 300 轮平均耗时 ± 标准差 | 测试 RMSE 均值 ± 标准差 |
|---|---|---:|---:|
| CPU | 当前主机 CPU | 3.065 ± 0.049 秒 | 6.739 ± 0.175 MPa |
| GPU | NVIDIA GeForce MX570 | 7.275 ± 0.254 秒 | 6.661 ± 0.093 MPa |

加速比定义为 CPU 平均耗时 ÷ GPU 平均耗时。本次约为 0.42，表示该小型网络在当前主机上的 CPU 训练约快 2.37 倍。这个测量只反映当前硬件、软件和模型规模。

## 独立预测

默认使用 5 条示例输入预测，并将结果保存至 `results/prediction_output.csv`：

```powershell
python src/predict.py
```

自备 CSV 需包含上述 8 个英文特征列；若还包含 `Concrete compressive strength`，输出会同时列出真实值和绝对误差。

```powershell
python src/predict.py .\data\my_mix.csv --output .\results\my_predictions.csv
```

预测单位为 MPa。脚本从权重文件加载训练得到的网络参数和标准化参数，因此预测时使用与训练一致的输入处理。

## 图表与结果文件

- `figures/training_curve.png`：基础 MLP 的训练/验证误差曲线。
- `figures/prediction_scatter.png`：测试集真实值与预测值散点图。
- `figures/ablation_rmse.png`：各消融配置的误差比较。
- `figures/ablation_val_curves_seed42.png`：同一训练种子下各配置的验证 RMSE 曲线。
- `figures/cpu_gpu_training_time.png`：CPU/GPU 训练耗时对比。
- `results/`：指标、逐次实验记录、模型权重及预测输出。

## 报告与复现

实验设计和现阶段结果说明见 [`实验与报告方案.md`](实验与报告方案.md)。ACM 双栏英文报告的 LaTeX 源文件、图表和 PDF 位于 [`report/`](report/)。本项目使用 **MiKTeX Portable 25.12** 编译。在本机从 `report` 目录的 PyCharm 终端运行以下命令两次，以更新交叉引用和目录信息（若便携版放在其他位置，请相应修改可执行文件路径）：

```powershell
& 'F:\miktex-portable\texmfs\install\miktex\bin\x64\pdflatex.exe' -interaction=nonstopmode -halt-on-error -jobname=concrete_strength main.tex
```

命令在 `report` 目录执行时，生成的文件为 `report/concrete_strength.pdf`。本项目已使用 MiKTeX Portable 25.12 完成双栏报告编译；若连续编译两次，第二次用于更新交叉引用。

## 忽略文件与版本管理

`.gitignore` 排除 Python 缓存、PyCharm 配置目录及虚拟环境。项目代码、数据、模型、实验结果和图表应按课程提交要求纳入版本管理。常用本地版本命令：

```bash
git status
git diff
git add README.md
git commit -m "完善项目 README"
git log --oneline
```

## License

UCI 数据集标注为 CC BY 4.0。项目代码的许可尚未单独指定；若要公开发布，请根据课程或个人要求补充代码许可证。
