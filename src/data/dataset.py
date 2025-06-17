# src/data/dataset.py
import os
import h5py
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("config/default.yaml")
PROC_PATH = cfg.data.processed_path
SPLITS    = cfg.data.splits
SOURCES   = ["drums", "bass", "other", "vocals"]

class AudioDataset(Dataset):
    def __init__(self, split: str, transform=None):
        assert split in SPLITS, f"Split must be one of {SPLITS}"
        h5_path = os.path.join(PROC_PATH, f"{split}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        self.h5f = h5py.File(h5_path, 'r')
        self.transform = transform

        # Build flat index of (track_id, segment_idx)
        self.index = []
        for track_id in self.h5f["mixture"]:
            n_seg = self.h5f["mixture"][track_id].shape[0]
            for i in range(n_seg):
                self.index.append((track_id, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        track_id, seg_i = self.index[idx]
        # Load mixture (float16 → float32 for model)
        mix_np = self.h5f["mixture"][track_id][seg_i]
        mix = torch.from_numpy(mix_np).float()  # shape (F, T)

        # Load targets and stack: (4, F, T)
        targets = []
        for src in SOURCES:
            arr = self.h5f[src][track_id][seg_i]
            targets.append(torch.from_numpy(arr).float())
        target = torch.stack(targets, dim=0)

        # Optionally apply a transform to mix
        if self.transform:
            mix = self.transform(mix)

        return mix, target
