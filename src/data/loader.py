import os
from glob import glob
import zipfile
import subprocess
import shutil
import logging
log = logging.getLogger(__name__)

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found on PATH—please install it first")

from omegaconf import OmegaConf

cfg = OmegaConf.load("config/default.yaml")
RAW_PATH = cfg.data.raw_path
SPLITS = cfg.data.splits

def extract_archives(raw_path: str):
    """
    Unzip train.zip → data/raw/train/ (flattening away the inner 'train/' folder)
          test.zip  → data/raw/test/   (same for 'test/')
    """
    for split in (SPLITS):
        zip_path = os.path.join(raw_path, f"{split}.zip")
        extract_dir = os.path.join(raw_path, split)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Missing archive: {zip_path}")

        # Only extract once
        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            log.info(f"Skipping extraction for {split} as {extract_dir} already exists.")
            continue

        log.info(f"Extracting {zip_path} → {extract_dir} (flattening top folder)…")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # skip macOS metadata and directories
                if member.startswith("__MACOSX/") or member.endswith("/"):
                    continue

                # strip off the first path component ("train/" or "test/")
                parts = member.split("/", 1)
                if len(parts) == 1:
                    # file was at root of the zip
                    rel_path = parts[0]
                else:
                    rel_path = parts[1]

                dest_path = os.path.join(extract_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # copy the file data
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        log.info(f"Extraction complete for {split}. Contents now in {extract_dir}")


# map stem index → output filename
STEM_MAP = {
    0: "mixture.wav",
    1: "drums.wav",
    2: "bass.wav",
    3: "other.wav",
    4: "vocals.wav",
}


def split_stems_to_wav(split_dir: str):
    """
    For every .mp4 in split_dir:
      • create a subfolder named after the file, with any ".stem" removed
      • dump 5 mono-wav stems (mixture, drums, bass, other, vocals)
      • delete the original .mp4
    """
    mp4_files = glob(os.path.join(split_dir, "*.mp4"))
    for mp4_path in mp4_files:
        # derive folder name and strip out any ".stem" suffix
        filename = os.path.basename(mp4_path)
        name_only = os.path.splitext(filename)[0]  # e.g. "track01.stem"
        clean_name = name_only.replace(".stem", "")  # e.g. "track01"
        out_folder = os.path.join(split_dir, clean_name)
        os.makedirs(out_folder, exist_ok=True)

        # split each channel into its own wav
        for idx, stem_name in STEM_MAP.items():
            wav_path = os.path.join(out_folder, stem_name)
            if not os.path.exists(wav_path):
                cmd = [
                    "ffmpeg",
                    "-y",  # overwrite if exists
                    "-i",
                    mp4_path,  # input file
                    "-map",
                    f"0:a:{idx}",  # select stem channel
                    "-ac",
                    "1",  # mono output
                    wav_path,
                ]
                subprocess.run(cmd, check=True)

        # remove the original mp4 file now that stems are saved
        os.remove(mp4_path)

        log.info(f"Processed {mp4_path} → {out_folder} with {len(STEM_MAP)} stems.")

if __name__ == "__main__":
    # 1) Make sure DVC has pulled the .zip files
    #    (you must run `dvc pull` beforehand)
    extract_archives(RAW_PATH)

    # 2) Unpack stems → wavs
    for split in (SPLITS):
        split_dir = os.path.join(RAW_PATH, split)
        split_stems_to_wav(split_dir)

    log.info("All stems extracted to wav format successfully.")