from __future__ import annotations

print("="*60)
print("TASK 2 TRAINING SCRIPT STARTED")
print("="*60)

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from attention_model import MultiModalEngagementModel
from augmentation import FrameAugmenter, pad_or_trim_sequence
from config_optimal import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from gaze_extractor import GazeEstimator
from synthetic_data import generate_synthetic_class_zero


CLASS_NAME_TO_LABEL = {
    "distracted": 0,
    "nominally_engaged": 1,
    "highly_engaged": 2,
    "disengaged": 3,
}

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        phase: str,
        augmenter: FrameAugmenter,
        gaze_estimator: GazeEstimator,
        synthetic_multiplier: int,
        use_gaze: bool,
    ) -> None:
        self.samples: List[Dict] = []
        for sample in samples:
            self.samples.append({**sample, "synthetic": False})
            if (
                sample["label"] == MODEL_CONFIG["distracted_class_index"]
                and synthetic_multiplier > 1
                and phase == "phase1"
            ):
                for _ in range(synthetic_multiplier - 1):
                    self.samples.append({**sample, "synthetic": True})
        self.phase = phase
        self.augmenter = augmenter
        self.gaze_estimator = gaze_estimator
        self.synthetic_multiplier = synthetic_multiplier
        self.use_gaze = use_gaze
        self.base_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(DATA_CONFIG["frame_size"]), transforms.ToTensor()]
        )
        self.normalize = transforms.Normalize(mean=MEAN_STD[0], std=MEAN_STD[1])
        self.video_cache: Dict[str, List[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: str) -> List[np.ndarray]:
        if path in self.video_cache:
            return self.video_cache[path]
        cap = cv2.VideoCapture(path)
        frames: List[np.ndarray] = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            resized = cv2.resize(frame, DATA_CONFIG["frame_size"], interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from {path}")
        self.video_cache[path] = frames
        return frames

    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        total = len(frames)
        target = MODEL_CONFIG["sequence_length"]
        if total <= target:
            idxs = np.linspace(0, total - 1, num=total, dtype=int).tolist()
            repeated = (idxs * ((target + total - 1) // total))[:target]
            return [frames[i] for i in repeated]
        else:
            step = total / target
            idxs = [int(i * step) for i in range(target)]
            return [frames[i] for i in idxs]

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        frames_bgr = self._load_video(sample["path"])
        selected_frames = self._sample_frames(frames_bgr)
        gaze_features = self.gaze_estimator.batch_extract(selected_frames)
        frame_tensors = []
        for frame in selected_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = self.base_transform(frame_rgb)
            if sample["synthetic"]:
                tensor_frame = generate_synthetic_class_zero([tensor_frame])[0]
            tensor_frame = self.augmenter.augment_frame(tensor_frame, sample["label"])
            frame_tensors.append(self.normalize(tensor_frame))
        frames_tensor = pad_or_trim_sequence(frame_tensors)
        if not self.use_gaze:
            gaze_features = torch.zeros_like(gaze_features)
        else:
            gaze_features = gaze_features
        return {
            "frames": frames_tensor,
            "gaze": gaze_features,
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    gaze = torch.stack([item["gaze"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {"frames": frames, "gaze": gaze, "label": labels}


def discover_samples(root: str) -> List[Dict]:
    samples = []
    for class_name, label in CLASS_NAME_TO_LABEL.items():
        class_dir = Path(root) / class_name
        if not class_dir.exists():
            continue
        for ext in DATA_CONFIG["video_ext"]:
            for video_path in class_dir.glob(f"**/*{ext}"):
                samples.append({"path": str(video_path), "label": label})
    if not samples:
        raise RuntimeError(f"No training samples found under {root}")
    return samples


def stratified_split(samples: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)
    train, val = [], []
    rng = random.Random(seed)
    for label, items in by_label.items():
        rng.shuffle(items)
        split_idx = max(1, int(len(items) * (1 - val_ratio)))
        train.extend(items[:split_idx])
        val.extend(items[split_idx:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_dataloaders(
    train_samples: List[Dict],
    val_samples: List[Dict],
    phase: str,
    augmenter: FrameAugmenter,
    gaze_estimator: GazeEstimator,
    synthetic_multiplier: int,
    use_gaze: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = VideoDataset(train_samples, phase, augmenter, gaze_estimator, synthetic_multiplier, use_gaze)
    val_dataset = VideoDataset(val_samples, phase, augmenter, gaze_estimator, 1, True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def compute_metrics(labels: List[int], predictions: List[int]) -> Dict[str, float]:
    num_classes = MODEL_CONFIG["num_classes"]
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
    for label, pred in zip(labels, predictions):
        conf_mat[label, pred] += 1
    total = conf_mat.sum()
    accuracy = conf_mat.trace() / total if total else 0.0
    per_class_acc = conf_mat.diagonal() / np.maximum(conf_mat.sum(axis=1), 1)
    f1_scores = []
    for cls in range(num_classes):
        tp = conf_mat[cls, cls]
        fp = conf_mat[:, cls].sum() - tp
        fn = conf_mat[cls, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "class0_accuracy": float(per_class_acc[0]) if len(per_class_acc) > 0 else 0.0,
        "confusion_matrix": conf_mat.tolist(),
    }
    return metrics


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            gaze = batch["gaze"].to(device)
            logits = model(frames, gaze)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
    return compute_metrics(labels, predictions)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)
        gaze = batch["gaze"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(frames, gaze)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["grad_clip_norm"])
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        if batch_count % 5 == 0:
            print(f"  Batch {batch_count}/{len(dataloader)} - Loss: {loss.item():.4f}")
    return total_loss / max(1, len(dataloader))


def train_phase(
    model: nn.Module,
    train_samples: List[Dict],
    val_samples: List[Dict],
    phase: str,
    epochs: int,
    device: torch.device,
    class_weights: torch.Tensor,
    augmenter: FrameAugmenter,
    gaze_estimator: GazeEstimator,
    use_gaze: bool,
    log_records: List[Dict],
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    print(f"\n{'='*60}")
    print(f"STARTING {phase.upper()}")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | Use Gaze: {use_gaze}")
    
    train_loader, val_loader = build_dataloaders(
        train_samples,
        val_samples,
        phase,
        augmenter,
        gaze_estimator,
        TRAINING_CONFIG["synthetic_multiplier_class0"],
        use_gaze,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], weight_decay=TRAINING_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_metrics = {"accuracy": 0.0}
    best_state: Dict[str, torch.Tensor] | None = None
    patience = TRAINING_CONFIG["early_stopping_patience"]
    wait = 0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        metrics.update({"phase": phase, "epoch_loss": loss, "epoch": epoch})
        log_records.append(metrics)
        
        print(f"Train Loss: {loss:.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val F1: {metrics['macro_f1']:.4f}")
        
        if metrics["accuracy"] > best_metrics.get("accuracy", 0.0):
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f"✓ New best accuracy: {metrics['accuracy']*100:.2f}%")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping (patience={patience})")
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"\n{phase.upper()} Complete - Best Acc: {best_metrics['accuracy']*100:.2f}%")
    return best_metrics, best_state


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_ensemble(output_dir: Path) -> List[Path]:
    print("="*60)
    print("TASK 2: MULTI-CLASS CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Discovering videos from: {DATA_CONFIG['train_root']}")
    
    samples = discover_samples(DATA_CONFIG["train_root"])
    
    print(f"Found {len(samples)} videos:")
    label_counts = {}
    for sample in samples:
        label_counts[sample['label']] = label_counts.get(sample['label'], 0) + 1
    for label, count in sorted(label_counts.items()):
        class_name = [k for k, v in CLASS_NAME_TO_LABEL.items() if v == label][0]
        print(f"  Class {label} ({class_name}): {count} videos")
    
    saved_paths: List[Path] = []
    log_records: List[Dict] = []
    
    for idx, seed in enumerate(TRAINING_CONFIG["random_seeds"]):
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL {idx+1}/{len(TRAINING_CONFIG['random_seeds'])} (seed={seed})")
        print(f"{'='*60}")
        
        set_seed(seed)
        train_samples, val_samples = stratified_split(samples, val_ratio=0.2, seed=seed)
        
        print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
        
        augmenter = FrameAugmenter()
        gaze_estimator = GazeEstimator()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {device}")
        
        model = MultiModalEngagementModel(
            cnn_backbone=MODEL_CONFIG["cnn_backbone"],
            lstm_hidden_size=MODEL_CONFIG["lstm_hidden_size"],
            lstm_num_layers=MODEL_CONFIG["lstm_num_layers"],
            lstm_dropout=MODEL_CONFIG["lstm_dropout"],
            feature_dim=MODEL_CONFIG["feature_dim"],
            num_classes=MODEL_CONFIG["num_classes"],
        )
        model.to(device)
        class_weights = torch.tensor(TRAINING_CONFIG["class_weights"], dtype=torch.float32)

        global_best_metrics = {"accuracy": 0.0}
        global_best_state: Dict[str, torch.Tensor] | None = None

        phase1_metrics, phase1_state = train_phase(
            model,
            train_samples,
            val_samples,
            phase="phase1",
            epochs=TRAINING_CONFIG["num_epochs_phase1"],
            device=device,
            class_weights=class_weights,
            augmenter=augmenter,
            gaze_estimator=gaze_estimator,
            use_gaze=False,
            log_records=log_records,
        )

        if phase1_metrics["accuracy"] > global_best_metrics["accuracy"]:
            global_best_metrics = phase1_metrics
            global_best_state = phase1_state

        phase2_metrics, phase2_state = train_phase(
            model,
            train_samples,
            val_samples,
            phase="phase2",
            epochs=TRAINING_CONFIG["num_epochs_phase2"],
            device=device,
            class_weights=class_weights,
            augmenter=augmenter,
            gaze_estimator=gaze_estimator,
            use_gaze=True,
            log_records=log_records,
        )

        if phase2_metrics["accuracy"] > global_best_metrics["accuracy"]:
            global_best_metrics = phase2_metrics
            global_best_state = phase2_state

        phase3_metrics, phase3_state = train_phase(
            model,
            train_samples,
            val_samples,
            phase="phase3",
            epochs=TRAINING_CONFIG["num_epochs_phase3"],
            device=device,
            class_weights=class_weights,
            augmenter=FrameAugmenter(),
            gaze_estimator=gaze_estimator,
            use_gaze=True,
            log_records=log_records,
        )

        if phase3_metrics["accuracy"] > global_best_metrics["accuracy"]:
            global_best_metrics = phase3_metrics
            global_best_state = phase3_state

        if global_best_state is not None:
            model.load_state_dict(global_best_state)

        best_metrics = global_best_metrics
        model_path = output_dir / f"model_seed{seed}.pt"
        save_model(model, model_path)
        saved_paths.append(model_path)
        gaze_estimator.close()
        
        print(f"\n{'='*60}")
        print(f"MODEL {idx+1} COMPLETE")
        print(f"{'='*60}")
        print(f"Best Accuracy: {best_metrics['accuracy']*100:.2f}%")
        print(f"Saved: {model_path}")
        
        log_records.append(
            {
                "seed": seed,
                "best_accuracy": best_metrics["accuracy"],
                "best_macro_f1": best_metrics["macro_f1"],
                "best_class0_accuracy": best_metrics["class0_accuracy"],
            }
        )
    
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"records": log_records}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved: {len(saved_paths)}")
    print(f"Log saved: {log_path}")
    
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train boosted ensemble for Task 2")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Task2_Visual_Multiclass/checkpoints",
        help="Directory to store trained models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = train_ensemble(output_dir)
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "config": {
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "batch_size": TRAINING_CONFIG["batch_size"],
            "sequence_length": MODEL_CONFIG["sequence_length"],
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
