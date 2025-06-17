# src/data/dataset.py

import os
import h5py
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.global_hydra import GlobalHydra

SOURCES = ["drums", "bass", "other", "vocals"]

class AudioDataset(Dataset):
    """
    Loads spectrogram segments from split.h5.
    Lazily opens the file in each worker to avoid pickling the handle.
    """
    def __init__(self, cfg, split: str, transform=None):
        self.cfg = cfg
        SPLITS = cfg.data.splits
        assert split in SPLITS, f"Split must be one of {SPLITS}"
        self.transform = transform

        # Determine the project root: Hydra’s original cwd or fallback
        try:
            root = get_original_cwd()
        except ValueError:
            root = os.getcwd()

        # Now construct the path to the HDF5 file
        proc_dir = os.path.join(root, cfg.data.processed_path)
        self.h5_path = os.path.join(proc_dir, f"{split}.h5")
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        # Build flat list of (track_id, segment_idx)
        self.index = []
        with h5py.File(self.h5_path, "r") as h5f:
            for track_id in h5f["mixture"].keys():
                n_seg = h5f["mixture"][track_id].shape[0]
                for i in range(n_seg):
                    self.index.append((track_id, i))

        # Each worker will open its own handle here
        self.h5f = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Lazy-open per worker
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, "r")

        track_id, seg_i = self.index[idx]

        # Mixture: float16 -> float32
        mix_np = self.h5f["mixture"][track_id][seg_i]
        mix = torch.from_numpy(mix_np).float()  # (F, T)

        # Targets: stack the 4 sources -> (4, F, T)
        targets = []
        for src in SOURCES:
            arr = self.h5f[src][track_id][seg_i]
            targets.append(torch.from_numpy(arr).float())
        target = torch.stack(targets, dim=0)

        # Apply transform only to the mixture
        if self.transform:
            mix = self.transform(mix)

        return mix, target
