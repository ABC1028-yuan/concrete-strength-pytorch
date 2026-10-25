# PyCharm 运行入口

完整的项目说明、环境准备、实验命令、结果和图表索引见项目根目录的 [`README.md`](README.md)。

## 在 PyCharm 中运行

1. 在 PyCharm 中打开 `concrete_strength` 项目，并确认项目解释器已安装 `requirements.txt` 中的依赖。
2. 从 PyCharm 底部打开“终端”，确认当前路径为项目根目录。
3. 运行基础实验：

   ```powershell
   python src/run_experiment.py
   ```

4. 运行独立预测：

   ```powershell
   python src/predict.py
   ```

5. 查看 `results/` 中的指标和预测结果，查看 `figures/` 中的图表。

多种子消融和 CPU/GPU 测速命令、现有结果解读及输入 CSV 格式请参阅 [`README.md`](README.md)。
