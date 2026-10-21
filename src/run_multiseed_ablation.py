"""Repeat the MLP baseline with 3 seeds and run one-factor ablations.

Run froma the project root:
    python src/run_multiseed_ablation.py

All variants use the same train/validation/test split (split seed 42). The
validation RMSE selects the epoch; test metrics are reported for comparison.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concrete.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TARGET = "Concrete compressive strength"
FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
SEEDS = (42, 43, 44)
SPLIT_SEED = 42
MAX_EPOCHS = 300
BATCH_SIZE = 32

# Change exactly one setting at a time relative to the baseline.
CONFIGS = [
    {"name": "baseline", "standardize": True, "hidden": (32, 16), "lr": 0.001},
    {"name": "no_standardization", "standardize": False, "hidden": (32, 16), "lr": 0.001},
    {"name": "hidden_16", "standardize": True, "hidden": (16,), "lr": 0.001},
    {"name": "hidden_64_32", "standardize": True, "hidden": (64, 32), "lr": 0.001},
    {"name": "lr_0.0005", "standardize": True, "hidden": (32, 16), "lr": 0.0005},
    {"name": "lr_0.002", "standardize": True, "hidden": (32, 16), "lr": 0.002},
]


def build_model(hidden):
    dims = [len(FEATURES), *hidden, 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def score(y_true, y_pred):
    return {
        "test_MAE_MPa": mean_absolute_error(y_true, y_pred),
        "test_RMSE_MPa": np.sqrt(mean_squared_error(y_true, y_pred)),
        "test_R2": r2_score(y_true, y_pred),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(frame, config, seed, split):
    X_train, X_val, X_test, y_train, y_val, y_test = split
    if config["standardize"]:
        scaler = StandardScaler().fit(X_train)
        x_train = scaler.transform(X_train)
        x_val = scaler.transform(X_val)
        x_test = scaler.transform(X_test)
    else:
        x_train = X_train.to_numpy()
        x_val = X_val.to_numpy()
        x_test = X_test.to_numpy()

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_np = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    set_seed(seed)
    model = build_model(config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_np)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    tx_val = torch.from_numpy(x_val)
    ty_val = y_val_np.ravel()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(tx_val).numpy().ravel()
        val_rmse = float(np.sqrt(mean_squared_error(ty_val, val_pred)))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(x_test)).numpy().ravel()
    return {
        "variant": config["name"], "seed": seed,
        "standardize": config["standardize"],
        "hidden_layers": "-".join(map(str, config["hidden"])),
        "learning_rate": config["lr"], "best_epoch": best_epoch,
        "best_val_RMSE_MPa": best_val_rmse,
        **score(y_test_np, test_pred),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    missing = [column for column in [*FEATURES, TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    X, y = frame[FEATURES], frame[TARGET]

    # Fixed 60/20/20 split shared by every configuration and training seed.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SPLIT_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SPLIT_SEED
    )
    split = (X_train, X_val, X_test, y_train, y_val, y_test)

    torch.set_num_threads(1)
    rows = []
    total = len(CONFIGS) * len(SEEDS)
    for index, config in enumerate(CONFIGS, 1):
        for seed in SEEDS:
            result = run_one(frame, config, seed, split)
            rows.append(result)
            print(
                f"[{len(rows)}/{total}] {config['name']} seed={seed}: "
                f"test RMSE={result['test_RMSE_MPa']:.3f} MPa, "
                f"MAE={result['test_MAE_MPa']:.3f} MPa, R2={result['test_R2']:.3f}"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail_path = RESULTS / "ablation_runs.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary = detail.groupby("variant", sort=False).agg(
        seeds=("seed", "count"),
        MAE_mean_MPa=("test_MAE_MPa", "mean"),
        MAE_std_MPa=("test_MAE_MPa", "std"),
        RMSE_mean_MPa=("test_RMSE_MPa", "mean"),
        RMSE_std_MPa=("test_RMSE_MPa", "std"),
        R2_mean=("test_R2", "mean"),
        R2_std=("test_R2", "std"),
        best_epoch_mean=("best_epoch", "mean"),
    ).reset_index()
    summary_path = RESULTS / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(summary))
    plt.bar(x, summary["RMSE_mean_MPa"],
            yerr=summary["RMSE_std_MPa"].fillna(0), capsize=4,
            color=["#536B8E" if name == "baseline" else "#8B6BD6"
                   for name in summary["variant"]])
    plt.xticks(x, summary["variant"], rotation=20, ha="right")
    plt.ylabel("Test RMSE (MPa), mean ± std across seeds")
    plt.title("Three-seed MLP baseline and one-factor ablations")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure_path = FIGURES / "ablation_rmse.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()

    print("\nSummary (mean and standard deviation across 3 seeds):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nDetailed runs: {detail_path}")
    print(f"Summary table: {summary_path}")
    print(f"Comparison chart: {figure_path}")


if __name__ == "__main__":
    main()
