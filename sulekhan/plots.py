import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.figure(figsize=(14, 8))

# Loss
plt.subplot(2,2,1)
plt.plot(df["epoch"], df["train_loss"], label="Train")
plt.plot(df["epoch"], df["val_loss"], label="Val")
plt.title("Loss")
plt.legend()
plt.grid()

# AbsRel
plt.subplot(2,2,2)
plt.plot(df["epoch"], df["train_abs_rel"], label="Train")
plt.plot(df["epoch"], df["val_abs_rel"], label="Val")
plt.title("AbsRel")
plt.legend()
plt.grid()

# RMSE
plt.subplot(2,2,3)
plt.plot(df["epoch"], df["train_rmse"], label="Train")
plt.plot(df["epoch"], df["val_rmse"], label="Val")
plt.title("RMSE")
plt.legend()
plt.grid()

# SI-RMSE
plt.subplot(2,2,4)
plt.plot(df["epoch"], df["train_si_rmse"], label="Train")
plt.plot(df["epoch"], df["val_si_rmse"], label="Val")
plt.title("SI-RMSE")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("full_training_curve.png", dpi=150)
plt.show()