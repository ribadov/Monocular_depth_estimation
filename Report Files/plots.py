import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

BASELINE_PATH = "baseline_training_log.csv"
BETTERNET_PATH = "jay_baseline_training_log.csv"
FINAL_PATH = "depth_cues_training_log.csv"

OUT_DIR = "unified_training_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

baseline = pd.read_csv(BASELINE_PATH)
betternet = pd.read_csv(BETTERNET_PATH)
final = pd.read_csv(FINAL_PATH)

# ============================================================
# CORE NORMALIZER
# ============================================================

def normalize_phase_logs(df, name):
    """
    Handles:
    - logs with 'phase' column (train/val split per epoch)
    - repeated epochs
    """
    if "phase" not in df.columns:
        return None

    df = df.copy()

    # Create real epoch index per phase
    df["epoch"] = df["epoch_index_zero_based"] if "epoch_index_zero_based" in df.columns else df["epoch"]

    train = df[df["phase"] == "train"].sort_values("epoch")
    val = df[df["phase"] == "val"].sort_values("epoch")

    # Aggregate per epoch (important fix!)
    train_agg = train.groupby("epoch").mean(numeric_only=True)
    val_agg = val.groupby("epoch").mean(numeric_only=True)

    epochs = sorted(set(train_agg.index).union(set(val_agg.index)))

    out = pd.DataFrame({"epoch": epochs, "model": name})

    metrics = ["loss", "abs_rel", "rmse", "si_rmse"]

    for m in metrics:
        out[f"train_{m}"] = [train_agg[m].get(e, np.nan) for e in epochs]
        out[f"val_{m}"] = [val_agg[m].get(e, np.nan) for e in epochs]

    return out


def normalize_epoch_logs(df, name):
    """
    Handles true epoch-based logs (Final model)
    """
    df = df.copy().sort_values("epoch")
    df["model"] = name
    return df


# ============================================================
# BUILD UNIFIED DATASETS
# ============================================================

baseline_df = normalize_phase_logs(baseline, "Baseline TinyUNet")
betternet_df = normalize_phase_logs(betternet, "BetterNet")
final_df = normalize_epoch_logs(final, "Final UNet")

all_models = [d for d in [baseline_df, betternet_df, final_df] if d is not None]

# ============================================================
# SAFE PLOT FUNCTION
# ============================================================

def plot_metric(metric, title, filename):
    plt.figure(figsize=(11, 6))

    for df in all_models:
        if f"train_{metric}" not in df.columns:
            continue

        plt.plot(df["epoch"], df[f"train_{metric}"], label=f"{df['model'].iloc[0]} - Train")
        plt.plot(df["epoch"], df[f"val_{metric}"], linestyle="--", label=f"{df['model'].iloc[0]} - Val")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()


# ============================================================
# MAIN PLOTS
# ============================================================

plot_metric("loss", "Loss Comparison (All UNet Models)", "loss.png")
plot_metric("abs_rel", "Abs Relative Error Comparison", "abs_rel.png")
plot_metric("rmse", "RMSE Comparison", "rmse.png")
plot_metric("si_rmse", "SI-RMSE Comparison", "si_rmse.png")

# ============================================================
# GENERALIZATION GAP
# ============================================================

plt.figure(figsize=(11, 6))

for df in all_models:
    if "train_loss" not in df.columns:
        continue

    gap = df["val_loss"] - df["train_loss"]
    plt.plot(df["epoch"], gap, label=df["model"].iloc[0])

plt.axhline(0, color="black", linewidth=1)
plt.title("Generalization Gap (Val Loss - Train Loss)")
plt.xlabel("Epoch")
plt.ylabel("Gap")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "generalization_gap.png"), dpi=300)
plt.close()

# ============================================================
# NORMALIZED CONVERGENCE
# ============================================================

plt.figure(figsize=(11, 6))

for df in all_models:
    if "val_loss" not in df.columns:
        continue

    v = df["val_loss"].values.astype(float)

    norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)

    plt.plot(df["epoch"], norm, label=df["model"].iloc[0])

plt.title("Normalized Validation Loss (Convergence Speed)")
plt.xlabel("Epoch")
plt.ylabel("Normalized Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=300)
plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("DONE ✔")
print("Saved to:", OUT_DIR)
print("""
Generated plots:
- loss.png
- abs_rel.png
- rmse.png
- si_rmse.png
- generalization_gap.png
- convergence.png
""")