import pandas as pd
import matplotlib.pyplot as plt

# POS rPPG CSV load karo
df = pd.read_csv("subject_10_Vid_6_pos.csv")

# Plot POS rPPG signal
plt.figure(figsize=(10,4))
plt.plot(df["rppg_signal"], color="green")
plt.title("Extracted rPPG Signal using POS Algorithm")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("subject_10_Vid_6_pos_plot.png", dpi=150)
print("✅ Plot saved to subject_10_Vid_6_pos_plot.png")
plt.show()
