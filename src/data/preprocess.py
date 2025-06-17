# src/data/preprocess.py
import os
import glob
import numpy as np
import h5py
import torch
from src.data.preprocess_utils import load_audio, compute_stft
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration
cfg = OmegaConf.load("config/default.yaml")
RAW_PATH    = cfg.data.raw_path        # e.g. "data/raw"
PROC_PATH   = cfg.data.processed_path  # e.g. "data/processed"
SPLITS      = cfg.data.splits          # ["train","test"]
SEG_LEN     = cfg.data.segment_length  # frames per segment
SOURCES     = ["mixture", "drums", "bass", "other", "vocals"]

class SpectrogramPreprocessor:
    def __init__(self):
        os.makedirs(PROC_PATH, exist_ok=True)

    def process_split(self, split: str):
        h5_path = os.path.join(PROC_PATH, f"{split}.h5")
        mode = 'a'  # append or create
        with h5py.File(h5_path, mode) as h5f:
            for source in SOURCES:
                grp = h5f.require_group(source)
                pattern = os.path.join(RAW_PATH, split, "*", f"{source}.wav")
                for wav_path in glob.glob(pattern):
                    track_id = os.path.basename(os.path.dirname(wav_path))
                    if track_id in grp:
                        log.info(f"Skipping existing {split}/{source}/{track_id}")
                        continue

                    log.info(f"Processing {split}/{source}/{track_id}")
                    wav  = load_audio(wav_path)
                    spec = compute_stft(wav)                      # (2, F, T)
                    # magnitude-only, float16
                    mag  = torch.sqrt(spec[0].square()+spec[1].square()) \
                              .half().numpy()                  # (F, T)

                    F, T = mag.shape
                    n_seg = (T - SEG_LEN) // SEG_LEN + 1
                    segments = np.stack([
                        mag[:, i*SEG_LEN:(i+1)*SEG_LEN] for i in range(n_seg)
                    ], axis=0)  # (n_seg, F, SEG_LEN)

                    # write compressed chunked dataset
                    dset = grp.create_dataset(
                        name=track_id,
                        data=segments,
                        dtype='float16',
                        compression='gzip',
                        chunks=(1, F, SEG_LEN)
                    )
                    log.info(f"Wrote {split}.h5 -> /{source}/{track_id} [{n_seg} segments]")

if __name__ == "__main__":
    pre = SpectrogramPreprocessor()
    for split in SPLITS:
        log.info(f"Starting split: {split}")
        pre.process_split(split)
    log.info("All splits processed. HDF5 files ready under data/processed/")
