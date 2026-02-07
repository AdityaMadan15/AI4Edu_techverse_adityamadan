import numpy as np
import os
import shutil

# Paths
TASK2_FEATURES_DIR = "../Task2_Fresh/features"
TASK4_DATA_DIR = "data"

def pack():
    print("Packing features for Task 4...")
    
    if not os.path.exists(TASK4_DATA_DIR):
        os.makedirs(TASK4_DATA_DIR)

    # Load files from Task 2
    try:
        features = np.load(os.path.join(TASK2_FEATURES_DIR, "features.npy"))
        labels = np.load(os.path.join(TASK2_FEATURES_DIR, "labels.npy"))
        
        names_path = os.path.join(TASK2_FEATURES_DIR, "video_names.npy")
        if not os.path.exists(names_path):
            print("Error: video_names.npy not found. Feature extraction incomplete/old.")
            return False
            
        video_names = np.load(names_path)
        
        # Strip extensions to get IDs
        video_ids = [os.path.splitext(f)[0] for f in video_names]
        
        # Save to npz
        output_path = os.path.join(TASK4_DATA_DIR, "features.npz")
        np.savez(output_path, features=features, labels=labels, video_ids=video_ids)
        print(f"Successfully created {output_path}")
        print(f"  Features: {features.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  IDs: {len(video_ids)}")
        return True
        
    except Exception as e:
        print(f"Error packing features: {e}")
        return False

if __name__ == "__main__":
    if pack():
        print("Ready for Task 4.")
    else:
        print("Failed.")
