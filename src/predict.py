"""Load the saved MLP and predict concrete compressive strength from a CSV.

The input CSV must contain the eight feature columns listed in the checkpoint.
If it also contains ``Concrete compressive strength``, the output includes the
actual value and absolute error for each row.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]
TARGET = "Concrete compressive strength"


def load_model(checkpoint_path: Path):
    # This is the project's own checkpoint, which stores numpy scaler arrays.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = nn.Sequential(
        nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    features = checkpoint.get("input_features", FEATURES)
    if features != FEATURES:
        raise ValueError(f"Checkpoint feature order does not match expected order: {features}")
    return model, mean, scale


def main():
    parser = argparse.ArgumentParser(description="Predict concrete strength (MPa) from a CSV.")
    parser.add_argument("input_csv", nargs="?", type=Path,
                        default=ROOT / "data" / "prediction_input_template.csv",
                        help="CSV containing the eight feature columns")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "mlp_best.pt",
                        help="saved MLP checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "prediction_output.csv",
                        help="where to save prediction results")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    source = pd.read_csv(args.input_csv)
    missing = [name for name in FEATURES if name not in source.columns]
    if missing:
        raise ValueError("Input CSV is missing required feature columns: " + ", ".join(missing))
    X = source[FEATURES].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Feature columns contain missing values; fill or remove them before prediction.")

    model, mean, scale = load_model(args.checkpoint)
    standardized = ((X.to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(standardized)).numpy().ravel()

    result = source[FEATURES].copy()
    if TARGET in source.columns:
        actual = pd.to_numeric(source[TARGET], errors="raise").to_numpy(dtype=float)
        result["Actual strength (MPa)"] = actual
        result["Predicted strength (MPa)"] = prediction
        result["Absolute error (MPa)"] = np.abs(actual - prediction)
    else:
        result["Predicted strength (MPa)"] = prediction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.6f", encoding="utf-8-sig")
    print(f"Predicted {len(result)} row(s); saved to: {args.output}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
