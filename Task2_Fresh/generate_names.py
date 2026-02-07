import os
import numpy as np
from config import Config

def generate_names():
    video_names = []
    
    print("Reconstructing video names from directory traversal...")
    print(f"Data Source: {Config.DATA_DIR}")
    
    # EXACT same logic as extract_features.py
    for class_name, label in Config.CLASS_MAP.items():
        class_dir = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Skipping {class_name}, directory not found.")
            continue
            
        # os.listdir order is critical here
        # We assume it hasn't changed since features.npy was generated
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                video_names.append(fname)
                
    print(f"Found {len(video_names)} videos.")
    
    output_path = os.path.join(Config.FEATURES_DIR, "video_names.npy")
    np.save(output_path, video_names)
    print(f"Saved names to {output_path}")

if __name__ == "__main__":
    generate_names()
