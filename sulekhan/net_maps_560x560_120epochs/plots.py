import pandas as pd
import matplotlib.pyplot as plt

# ---- Load CSV ----
file_path = "training_log.csv"
df = pd.read_csv(file_path)

# ---- Create continuous epoch index ----
df["epoch_cont"] = range(len(df))

epochs = df["epoch_cont"].values
history = df

plt.figure(figsize=(14, 8))

# ---- Loss ----
plt.subplot(2, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# ---- AbsRel ----
plt.subplot(2, 2, 2)
plt.plot(epochs, history["train_abs_rel"], label="Train AbsRel")
plt.plot(epochs, history["val_abs_rel"], label="Val AbsRel")
plt.title("Absolute Relative Error")
plt.xlabel("Epoch")
plt.ylabel("AbsRel")
plt.legend()
plt.grid(True)

# ---- RMSE ----
plt.subplot(2, 2, 3)
plt.plot(epochs, history["train_rmse"], label="Train RMSE")
plt.plot(epochs, history["val_rmse"], label="Val RMSE")
plt.title("RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

# ---- SI-RMSE ----
plt.subplot(2, 2, 4)
plt.plot(epochs, history["train_si_rmse"], label="Train SI-RMSE")
plt.plot(epochs, history["val_si_rmse"], label="Val SI-RMSE")
plt.title("Scale-Invariant RMSE")
plt.xlabel("Epoch")
plt.ylabel("SI-RMSE")
plt.legend()
plt.grid(True)

plt.tight_layout()

# ---- Save ----
plt.savefig("training_curves_full.png", dpi=150)

plt.show()