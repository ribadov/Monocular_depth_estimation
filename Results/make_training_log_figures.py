#!/usr/bin/env python3
"""
Create publication-ready figures from the three training logs.

Expected files in the same directory as this script, or pass --input-dir:
  - training_log_SmallUNet.csv
  - training_log_MediumUNet.csv
  - training_log_DepthModelWithCues.csv

Outputs are written to --out-dir, default: figures_training_logs/
  - fig_validation_curves.pdf/.png
  - fig_best_metrics.pdf/.png
  - fig_generalization_gap.pdf/.png
  - fig_runtime_tradeoff.pdf/.png
  - main_results_table.tex
  - training_log_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOGS: Dict[str, str] = {
    "SmallUNet": "training_log_SmallUNet.csv",
    "MediumUNet": "training_log_MediumUNet.csv",
    "DepthModelWithCues": "training_log_DepthModelWithCues.csv",
}

DISPLAY_NAMES: Dict[str, str] = {
    "SmallUNet": "SmallUNet\n(baseline)",
    "MediumUNet": "MediumUNet",
    "DepthModelWithCues": "DepthModel\n+ Cues",
}

METRICS: List[Tuple[str, str]] = [
    ("loss", "Validation loss"),
    ("rmse", "RMSE"),
    ("si_rmse", "SI-RMSE"),
    ("abs_rel", "AbsRel"),
]

# Use a fixed palette so the same model has the same appearance in every figure.
COLORS: Dict[str, str] = {
    "SmallUNet": "#7f7f7f",
    "MediumUNet": "#1f77b4",
    "DepthModelWithCues": "#2ca02c",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_logs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    logs: Dict[str, pd.DataFrame] = {}
    for model, filename in LOGS.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected log file: {path}")

        df = pd.read_csv(path)
        required = {"epoch", "phase", "loss", "rmse", "si_rmse", "abs_rel", "epoch_elapsed_sec"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")

        # Standardize phase labels and sort just in case.
        df["phase"] = df["phase"].astype(str).str.lower()
        df = df.sort_values(["epoch", "phase"]).reset_index(drop=True)
        logs[model] = df
    return logs


def one_row_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per epoch for epoch-level metadata such as elapsed time."""
    return df.drop_duplicates(subset=["epoch"], keep="first").sort_values("epoch")


def summarize(logs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in logs.items():
        train = df[df["phase"] == "train"].copy()
        val = df[df["phase"] == "val"].copy()
        if train.empty or val.empty:
            raise ValueError(f"{model} must contain both train and val rows.")

        row = {
            "model": model,
            "img_size": int(df["img_size"].iloc[0]) if "img_size" in df.columns else np.nan,
            "batch_size": int(df["batch_size"].iloc[0]) if "batch_size" in df.columns else np.nan,
            "train_samples": int(df["train_dataset_size"].iloc[0]) if "train_dataset_size" in df.columns else np.nan,
            "val_samples": int(df["val_dataset_size"].iloc[0]) if "val_dataset_size" in df.columns else np.nan,
            "avg_epoch_sec": one_row_per_epoch(df)["epoch_elapsed_sec"].mean(),
            "total_time_h": one_row_per_epoch(df)["epoch_elapsed_sec"].sum() / 3600.0,
            "final_train_loss": float(train.iloc[-1]["loss"]),
            "final_val_loss": float(val.iloc[-1]["loss"]),
        }
        row["final_loss_gap"] = row["final_val_loss"] - row["final_train_loss"]

        for metric, _ in METRICS:
            best_idx = val[metric].idxmin()
            row[f"best_{metric}"] = float(val.loc[best_idx, metric])
            row[f"epoch_best_{metric}"] = int(val.loc[best_idx, "epoch"])
            row[f"final_val_{metric}"] = float(val.iloc[-1][metric])

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(summary["model"], categories=list(LOGS.keys()), ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(logs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, METRICS):
        for model, df in logs.items():
            val = df[df["phase"] == "val"]
            ax.plot(
                val["epoch"],
                val[metric],
                label=DISPLAY_NAMES[model].replace("\n", " "),
                color=COLORS[model],
                linewidth=2.0,
            )
            best_idx = val[metric].idxmin()
            best_epoch = val.loc[best_idx, "epoch"]
            best_value = val.loc[best_idx, metric]
            ax.scatter(best_epoch, best_value, color=COLORS[model], s=20, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_xlim(left=1)

    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Validation curves across model variants", y=1.02, fontsize=13)
    fig.tight_layout()
    save_all(fig, out_dir, "fig_validation_curves")


def plot_best_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    axes = axes.ravel()
    x = np.arange(len(summary))
    labels = [DISPLAY_NAMES[m] for m in summary["model"].astype(str)]

    for ax, (metric, title) in zip(axes, METRICS):
        values = summary[f"best_{metric}"].to_numpy()
        bar_colors = [COLORS[m] for m in summary["model"].astype(str)]
        bars = ax.bar(x, values, color=bar_colors, width=0.65)
        ax.set_title(f"Best {title}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(title)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_ylim(top=max(values) * 1.18)

    fig.suptitle("Best validation metrics, lower is better", y=1.02, fontsize=13)
    fig.tight_layout()
    save_all(fig, out_dir, "fig_best_metrics")


def plot_generalization_gap(logs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    summary = summarize(logs)
    x = np.arange(len(summary))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    train_bars = ax.bar(
        x - width / 2,
        summary["final_train_loss"],
        width,
        label="Final train loss",
        color="#9ecae1",
    )
    val_bars = ax.bar(
        x + width / 2,
        summary["final_val_loss"],
        width,
        label="Final val loss",
        color="#3182bd",
    )

    for i, gap in enumerate(summary["final_loss_gap"]):
        ax.annotate(
            f"gap={gap:.3f}",
            xy=(x[i], max(summary.loc[i, "final_train_loss"], summary.loc[i, "final_val_loss"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in summary["model"].astype(str)])
    ax.set_ylabel("Loss")
    ax.set_title("Final train-validation loss gap")
    ax.legend(frameon=True)
    ax.bar_label(train_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(val_bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    save_all(fig, out_dir, "fig_generalization_gap")


def plot_runtime_tradeoff(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.2))

    for _, row in summary.iterrows():
        model = str(row["model"])
        ax.scatter(
            row["total_time_h"],
            row["best_loss"],
            s=max(80, float(row["img_size"]) * 0.75),
            color=COLORS[model],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            DISPLAY_NAMES[model].replace("\n", " "),
            xy=(row["total_time_h"], row["best_loss"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Total training time (hours)")
    ax.set_ylabel("Best validation loss")
    ax.set_title("Accuracy-runtime trade-off")
    ax.invert_yaxis()  # lower loss is better, so better models appear higher.
    fig.tight_layout()
    save_all(fig, out_dir, "fig_runtime_tradeoff")


def write_latex_table(summary: pd.DataFrame, out_dir: Path) -> None:
    # A compact table suitable for direct inclusion with \input{figures_training_logs/main_results_table.tex}
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Best validation metrics for the three model variants. Lower is better for all metrics.}")
    lines.append(r"    \label{tab:main_results}")
    lines.append(r"    \resizebox{\linewidth}{!}{%")
    lines.append(r"    \begin{tabular}{lcccccc}")
    lines.append(r"        \toprule")
    lines.append(r"        Model & Img. size & Val. loss & RMSE & SI-RMSE & AbsRel & Time (h) \\")
    lines.append(r"        \midrule")

    for _, row in summary.iterrows():
        name = str(row["model"])
        if name == "SmallUNet":
            latex_name = r"SmallUNet"
        elif name == "MediumUNet":
            latex_name = r"MediumUNet"
        else:
            latex_name = r"DepthModelWithCues"

        lines.append(
            "        "
            + f"{latex_name} & "
            + f"{int(row['img_size'])} & "
            + f"{row['best_loss']:.4f} & "
            + f"{row['best_rmse']:.4f} & "
            + f"{row['best_si_rmse']:.4f} & "
            + f"{row['best_abs_rel']:.4f} & "
            + f"{row['total_time_h']:.2f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    (out_dir / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training logs for SmallUNet, MediumUNet, and DepthModelWithCues.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing the CSV logs.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures_training_logs"), help="Directory for output figures/tables.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    logs = load_logs(args.input_dir)
    summary = summarize(logs)
    summary.to_csv(args.out_dir / "training_log_summary.csv", index=False)

    plot_validation_curves(logs, args.out_dir)
    plot_best_metrics(summary, args.out_dir)
    plot_generalization_gap(logs, args.out_dir)
    plot_runtime_tradeoff(summary, args.out_dir)
    write_latex_table(summary, args.out_dir)

    print(f"Wrote figures and tables to: {args.out_dir.resolve()}")
    print(summary[["model", "best_loss", "best_rmse", "best_si_rmse", "best_abs_rel", "total_time_h", "final_loss_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
