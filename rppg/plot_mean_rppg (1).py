import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all three rPPG signals
df_chrom = pd.read_csv("subject_10_Vid_6.csv")
df_pos = pd.read_csv("subject_10_Vid_6_pos.csv")
df_physnet = pd.read_csv("subject_10_Vid_6_physnet.csv")

# Get minimum length to align all signals
min_len = min(len(df_chrom), len(df_pos), len(df_physnet))

# Trim all signals to same length
chrom_signal = df_chrom["rppg_signal"].values[:min_len]
pos_signal = df_pos["rppg_signal"].values[:min_len]
physnet_signal = df_physnet["rppg_signal"].values[:min_len]

# Calculate mean of all three signals
mean_signal = (chrom_signal + pos_signal + physnet_signal) / 3

# Create DataFrame for saving
df_mean = pd.DataFrame({
    "frame": np.arange(min_len),
    "rppg_signal": mean_signal
})

# Save the mean signal
df_mean.to_csv("subject_10_Vid_6_mean.csv", index=False)

# Plot the mean signal
plt.figure(figsize=(12, 5))
plt.plot(df_mean["frame"], df_mean["rppg_signal"], linewidth=1.5, color='purple', alpha=0.8)
plt.title("Mean rPPG Signal (CHROM + POS + PhysNet)", fontsize=14, fontweight='bold')
plt.xlabel("Frame Index", fontsize=12)
plt.ylabel("Normalized Amplitude", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("subject_10_Vid_6_mean_plot.png", dpi=150)
print("✅ Mean signal saved to subject_10_Vid_6_mean.csv")
print("✅ Plot saved to subject_10_Vid_6_mean_plot.png")
plt.show()
