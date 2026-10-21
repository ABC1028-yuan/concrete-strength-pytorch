# 混凝土抗压强度预测项目（PyCharm）

## 已完成内容

- `src/run_experiment.py`：从 UCI 原始数据训练线性回归基线和 MLP，并保存模型、指标及图表。
- `src/predict.py`：加载训练好的 MLP，对 CSV 配方预测强度；若输入含真实强度列，也输出绝对误差。
- `data/prediction_input_template.csv`：可直接运行的 5 条示例输入（含真实强度用于误差展示）。复制后替换为新配方，即可预测新的样本。

## 在 PyCharm 中运行

1. 确认项目解释器已安装 `requirements.txt` 中的依赖。
2. 在项目根目录打开终端，先训练（已有 `results/mlp_best.pt` 时可跳过）：

   ```powershell
   python src/run_experiment.py
   ```

3. 使用示例数据预测：

   ```powershell
   python src/predict.py
   ```

4. 预测自己的 CSV：CSV 必须含有这 8 个英文列名：`Cement`、`Blast Furnace Slag`、`Fly Ash`、`Water`、`Superplasticizer`、`Coarse Aggregate`、`Fine Aggregate`、`Age`。

   ```powershell
   python src/predict.py .\data\my_mix.csv --output .\results\my_predictions.csv
   ```

输出默认保存到 `results/prediction_output.csv`，预测单位为 MPa。模型使用训练集拟合的标准化参数，因此预测脚本会自动复用保存于权重文件中的参数。

## 多随机种子与消融实验

在 PyCharm Terminal 的项目根目录运行：

```powershell
python src/run_multiseed_ablation.py
```

脚本固定 60%/20%/20% 的数据划分（划分种子 42），以训练种子 42、43、44 重复六种配置：基础 MLP、关闭输入标准化、单隐藏层、加宽网络，以及两档学习率。每项只相对基础配置改变一个因素；每轮依据验证集 RMSE 选最佳轮次，再计算测试集 MAE、RMSE、R²。完整运行会训练 18 次，结果保存到 `results/ablation_runs.csv`（逐次记录）和 `results/ablation_summary.csv`（均值与标准差），图保存为 `figures/ablation_rmse.png`。

## CPU 与 GPU 训练速度对比

当前 PyCharm 解释器检测到 NVIDIA GeForce MX570，CUDA 可用。运行：

```powershell
python src/benchmark_cpu_gpu.py
```

默认对同一划分、同一 MLP、同一 batch size 和学习率，各运行 CPU/GPU 300 轮并重复 3 次；正式计时前各做 5 轮预热，并在 GPU 计时前后同步 CUDA。计时从数据已放到设备后开始，反映训练循环时间，不含一次性数据传输。结果保存到 `results/cpu_gpu_benchmark_runs.csv`、`results/cpu_gpu_benchmark_summary.csv` 和 `figures/cpu_gpu_training_time.png`。网络较小，GPU 不一定更快，实际结论以本机测量为准。

本机一次正式测量结果（300 轮，3 次）：CPU 平均 3.065 秒（标准差 0.049），MX570 GPU 平均 7.275 秒（标准差 0.254）；该小网络的 CPU 约快 2.37 倍。两设备测试 RMSE 均值分别为 6.739 和 6.661 MPa，接近。该速度结论只适用于当前硬件、PyTorch 环境和小规模模型。
Test line for revert.
