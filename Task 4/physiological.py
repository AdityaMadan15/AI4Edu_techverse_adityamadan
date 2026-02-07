import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, welch

def extract_rppg_features(csv_path, fs=30):
    # Check if file exists
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        signal = df["rppg_signal"].values
    except Exception as e:
        return None

    # --- HR Proxy ---
    peaks, _ = find_peaks(signal, distance=fs//2)
    if len(peaks) < 2:
        return None

    rr_intervals = np.diff(peaks) / fs
    hr = 60 / rr_intervals

    hr_mean = np.mean(hr)
    hr_std = np.std(hr)

    # --- HRV ---
    hrv_sdnn = np.std(rr_intervals)

    # --- SQI ---
    sqi = np.var(signal) / (np.var(np.diff(signal)) + 1e-6)

    # --- Frequency Domain ---
    freqs, psd = welch(signal, fs=fs)

    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)

    lf_power = np.sum(psd[lf_band])
    hf_power = np.sum(psd[hf_band]) + 1e-6
    lf_hf_ratio = lf_power / hf_power

    return np.array([
        hr_mean,
        hr_std,
        hrv_sdnn,
        sqi,
        lf_hf_ratio
    ])
