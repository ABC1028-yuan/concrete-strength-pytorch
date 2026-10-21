"""Benchmark identical MLP training on CPU and (when available) CUDA GPU.

Run from the project root:
    python src/benchmark_cpu_gpu.py

The default is 300 epochs and 3 measured repetitions per device. Timing starts
after the tensors are placed on the device and excludes the one-time transfer;
CUDA synchronization brackets the timed section. A short warm-up is excluded.
"""
from pathlib import Path
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concrete.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
TARGET = "Concrete compressive strength"
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def model_factory():
    return nn.Sequential(
        nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_for_epochs(x_train, y_train, x_test, y_test, device, epochs, seed, timed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Transfer happens before timing for both devices; report training compute time.
    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
    xte = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    yte = torch.as_tensor(y_test, dtype=torch.float32, device=device).reshape(-1)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    n = len(xtr)

    synchronize(device)
    start = time.perf_counter() if timed else None
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start_idx in range(0, n, BATCH_SIZE):
            indices = order[start_idx:start_idx + BATCH_SIZE]
            xb, yb = xtr[indices], ytr[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - start if timed else None

    model.eval()
    with torch.no_grad():
        pred = model(xte).reshape(-1)
        test_rmse = torch.sqrt(torch.mean((pred - yte) ** 2)).item()
    return elapsed, test_rmse


def main():
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA training speed.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="measured training epochs per run (default: 300)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="measured repetitions per device (default: 3)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="unmeasured warm-up epochs per device (default: 5)")
    args = parser.parse_args()
    if min(args.epochs, args.repeats) < 1 or args.warmup_epochs < 0:
        parser.error("epochs and repeats must be positive; warmup-epochs cannot be negative")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH)
    missing = [name for name in [*FEATURES, TARGET] if name not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42
    )
    # Match the baseline preprocessing: fit only on the training partition.
    scaler = StandardScaler().fit(X_train)
    x_train = scaler.transform(X_train).astype(np.float32)
    x_test = scaler.transform(X_test).astype(np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    torch.set_num_threads(1)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    else:
        print("CUDA GPU is unavailable in this Python environment; CPU will be measured only.")

    rows = []
    for device in devices:
        attempts = 2 if device.type == "cuda" else 1
        for attempt in range(attempts):
            try:
                # Warm up framework/device kernels before measured runs.
                if args.warmup_epochs:
                    train_for_epochs(x_train, y_train_np, x_test, y_test_np,
                                     device, args.warmup_epochs, seed=7, timed=False)
                device_rows = []
                for repeat in range(1, args.repeats + 1):
                    elapsed, rmse = train_for_epochs(
                        x_train, y_train_np, x_test, y_test_np,
                        device, args.epochs, seed=41 + repeat, timed=True,
                    )
                    device_rows.append({
                        "device": "GPU" if device.type == "cuda" else "CPU",
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        "repeat": repeat, "epochs": args.epochs,
                        "training_seconds": elapsed, "test_RMSE_MPa": rmse,
                    })
                rows.extend(device_rows)
                for row in device_rows:
                    print(
                        f"{row['device']} run {row['repeat']}/{args.repeats}: "
                        f"{row['training_seconds']:.3f}s, "
                        f"test RMSE={row['test_RMSE_MPa']:.3f} MPa"
                    )
                break
            except RuntimeError as exc:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if attempt + 1 < attempts:
                    print("CUDA initialization/training failed once; cleared the CUDA cache and retrying.")
                else:
                    print("GPU benchmark skipped after two CUDA failures.")
                    print("Other programs may be using the GPU; close them and run this script again.")
                    print(f"CUDA detail: {exc}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail_path = RESULTS / "cpu_gpu_benchmark_runs.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    summary = detail.groupby("device", sort=False).agg(
        device_name=("device_name", "first"), repeats=("repeat", "count"),
        epochs=("epochs", "first"),
        mean_seconds=("training_seconds", "mean"),
        std_seconds=("training_seconds", "std"),
        mean_test_RMSE_MPa=("test_RMSE_MPa", "mean"),
        std_test_RMSE_MPa=("test_RMSE_MPa", "std"),
    ).reset_index()
    summary_path = RESULTS / "cpu_gpu_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(summary["device"], summary["mean_seconds"],
            yerr=summary["std_seconds"].fillna(0), capsize=5,
            color=["#536B8E" if d == "CPU" else "#8B6BD6" for d in summary["device"]])
    plt.ylabel(f"Training time for {args.epochs} epochs (seconds)")
    plt.title("CPU vs GPU MLP training time (mean ± std)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure_path = FIGURES / "cpu_gpu_training_time.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()

    if {"CPU", "GPU"}.issubset(set(summary["device"])):
        cpu_time = float(summary.loc[summary.device == "CPU", "mean_seconds"].iloc[0])
        gpu_time = float(summary.loc[summary.device == "GPU", "mean_seconds"].iloc[0])
        print(f"\nGPU speedup (CPU mean time / GPU mean time): {cpu_time / gpu_time:.3f}x")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nDetailed runs: {detail_path}")
    print(f"Summary table: {summary_path}")
    print(f"Training-time chart: {figure_path}")


if __name__ == "__main__":
    main()
