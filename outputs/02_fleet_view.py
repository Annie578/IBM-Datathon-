import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

file_path = os.path.join("outputs", "fleet.csv")
fleet_df = pd.read_csv(file_path)

fleet_df = fleet_df.sort_values(by="predicted_rul", ascending=True).reset_index(
    drop=True
)

plt.figure(figsize=(14, 6))
colors = [
    "red" if rul < 20 else "orange" if rul < 50 else "navy"
    for rul in fleet_df["predicted_rul"]
]

bars = plt.bar(
    range(len(fleet_df)),
    fleet_df["predicted_rul"],
    color=colors,
    edgecolor="none",
)

plt.axhline(
    y=20, color="red", linestyle="--", linewidth=1.5, label="Critical Threshold (RUL < 20)"
)
plt.xlabel("Engine Index (Sorted by Risk)", fontsize=12)
plt.ylabel("Predicted Remaining Useful Life (Cycles)", fontsize=12)
plt.title(
    "Fleet Engine Ranking by Predicted RUL (Lowest First)", fontsize=14, pad=15
)
plt.legend()

# Show only every 5th engine ID on the x-axis to keep it clean
step = max(1, len(fleet_df) // 20)
plt.xticks(
    range(0, len(fleet_df), step),
    fleet_df["unit"].iloc[::step],
    rotation=0,
    fontsize=10,
)

plt.tight_layout()
plt.savefig("fleet_rul_ranking.png", dpi=300)
plt.show()