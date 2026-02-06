import pandas as pd
import matplotlib.pyplot as plt

# Load rPPG CSV file
df = pd.read_csv("data/rppg_signals/subject_10_Vid_6.csv")

# Plot signal
plt.figure(figsize=(10,4))
plt.plot(df["rppg_signal"])
plt.title("Extracted rPPG Signal using CHROM Algorithm")
plt.xlabel("Frame Index")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.show()
