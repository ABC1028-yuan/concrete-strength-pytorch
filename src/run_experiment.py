"""Read the downloaded UCI Concrete dataset and run a fixed-seed CPU baseline."""
from pathlib import Path
import json
import platform
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
for folder in (DATA, RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


def metrics(y, pred):
    return {"MAE_MPa": mean_absolute_error(y, pred),
            "RMSE_MPa": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred)}


def main():
    raw_xls = DATA / "raw" / "Concrete_Data.xls"
    if not raw_xls.exists():
        raise FileNotFoundError(f"UCI source file not found: {raw_xls}")
    frame = pd.read_excel(raw_xls)
    # The UCI workbook documents these nine columns in this order.
    frame.columns = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water",
                     "Superplasticizer", "Coarse Aggregate", "Fine Aggregate",
                     "Age", "Concrete compressive strength"]
    X = frame.iloc[:, :8].copy()
    y = frame.iloc[:, 8].copy()
    frame.to_csv(DATA / "concrete.csv", index=False, float_format="%.12g")
    frame.head(5).to_csv(DATA / "sample_preview.csv", index=False,
                         float_format="%.12g")
    profile = frame.describe().T
    profile.index.name = "field"
    profile.to_csv(RESULTS / "dataset_profile.csv", float_format="%.8g")
    print(f"shape={frame.shape}; missing={int(frame.isna().sum().sum())}")
    print("columns:", list(frame.columns))
    print("first 5 rows:\n", frame.head().to_string(index=False))
    print("sample input:\n", X.iloc[0].to_string())
    print(f"sample target: {float(y.iloc[0]):.3f} MPa")

    # Split 80/20, then 75/25 of the remaining 80% => 60/20/20 overall.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xv, Xte = (scaler.transform(a).astype(np.float32)
                    for a in (X_train, X_val, X_test))
    ytr = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    yv = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)
    yte = y_test.to_numpy(dtype=np.float32)

    lin = LinearRegression().fit(Xtr, ytr.ravel())
    linear_metrics = metrics(yte, lin.predict(Xte))

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16),
                          nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                              torch.from_numpy(ytr)), batch_size=32,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_x, val_y = torch.from_numpy(Xv), torch.from_numpy(yv)
    history, best_rmse, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(1, 301):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.from_numpy(Xtr)).numpy().ravel()
            val_pred = model(val_x).numpy().ravel()
        train_rmse = float(np.sqrt(mean_squared_error(ytr.ravel(), train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(yv.ravel(), val_pred)))
        history.append({"epoch": epoch, "train_RMSE_MPa": train_rmse,
                        "val_RMSE_MPa": val_rmse})
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_, "input_features": list(X.columns),
                "target_name": "Concrete compressive strength",
                "seed": seed, "best_epoch": best_epoch,
                "best_val_rmse_mpa": best_rmse}, RESULTS / "mlp_best.pt")
    model.eval()
    with torch.no_grad():
        mlp_pred = model(torch.from_numpy(Xte)).numpy().ravel()
    mlp_metrics = metrics(yte, mlp_pred)
    pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)
    pd.DataFrame([{"model": "LinearRegression", **linear_metrics},
                  {"model": "MLP_8_32_16_1", **mlp_metrics,
                   "best_val_RMSE_MPa": best_rmse,
                   "best_epoch": best_epoch}]).to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame({"actual_MPa": yte, "linear_prediction_MPa": lin.predict(Xte),
                  "mlp_prediction_MPa": mlp_pred}).to_csv(RESULTS / "test_predictions.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, 301), [h["train_RMSE_MPa"] for h in history], label="Train")
    plt.plot(range(1, 301), [h["val_RMSE_MPa"] for h in history], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("RMSE (MPa)")
    plt.title("MLP training and validation RMSE"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "training_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yte, mlp_pred, s=22, alpha=.7)
    bounds = [min(yte.min(), mlp_pred.min()), max(yte.max(), mlp_pred.max())]
    plt.plot(bounds, bounds, "k--", linewidth=1)
    plt.xlabel("Actual strength (MPa)"); plt.ylabel("Predicted strength (MPa)")
    plt.title("MLP predictions on test set"); plt.tight_layout()
    plt.savefig(FIGURES / "prediction_scatter.png", dpi=180); plt.close()
    metadata = {
        "source_file": "data/raw/Concrete_Data.xls",
        "dataset_id": 165,
        "n_samples": int(len(frame)), "n_features": int(X.shape[1]),
        "split_seed": 42, "training_seed": seed,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "model": "MLP 8-32-16-1 with ReLU",
        "optimizer": "Adam", "learning_rate": 0.001,
        "batch_size": 32, "max_epochs": 300,
        "best_epoch": best_epoch, "best_validation_rmse_mpa": best_rmse,
        "device": "cpu", "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "torch": torch.__version__, "scikit_learn": __import__("sklearn").__version__,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("split sizes:", len(X_train), len(X_val), len(X_test))
    print(pd.read_csv(RESULTS / "metrics.csv").to_string(index=False))
    print("Saved source-derived CSV, sample preview, profile, checkpoint, metrics, metadata and figures.")

if __name__ == "__main__":
    main()
