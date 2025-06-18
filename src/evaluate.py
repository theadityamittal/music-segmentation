#!/usr/bin/env python
# evaluate.py

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import h5py
from omegaconf import OmegaConf
from mir_eval.separation import bss_eval_sources
from src.models.unet import UNet

def load_model(checkpoint_path: str, cfg):
    device = torch.device("mps" if torch.mps else "cpu")
    model = UNet(
        in_ch=1,
        num_sources=len(cfg.data.sources) - 1,
        chans=cfg.model.chans,
        num_pool_layers=cfg.model.num_pool_layers
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def separate_track(mix_wav, model, device, cfg):
    wav, sr = librosa.load(mix_wav, sr=cfg.data.sample_rate, mono=True)
    stft = librosa.stft(wav, n_fft=cfg.data.n_fft, hop_length=cfg.data.hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    F, T = mag.shape
    SEG = cfg.data.segment_length
    pad = (SEG - (T % SEG)) % SEG
    if pad > 0:
        mag   = np.pad(mag,   ((0,0),(0,pad)), mode="constant")
        phase = np.pad(phase, ((0,0),(0,pad)), mode="constant")
    n_seg = mag.shape[1] // SEG

    preds = []
    with torch.no_grad():
        for i in range(n_seg):
            mseg = mag[:, i*SEG:(i+1)*SEG]
            x = torch.from_numpy(mseg).unsqueeze(0).unsqueeze(0).to(device).float()
            y = model(x)  # (1, S, F, SEG)
            preds.append(y.squeeze(0).cpu().numpy())
    # After concatenating your predictions to shape (S, F, T_pad)
    pred_mag = np.concatenate(preds, axis=2)

    # Trim to original length T_orig
    pred_mag = pred_mag[:, :, :T]      # (S, F, T_orig)
    phase    = phase[:, :T]            # trim phase to (F, T_orig)

    # 4) Reconstruct each source using mixture phase
    est_wavs = []
    for s in range(pred_mag.shape[0]):
        complex_spec = pred_mag[s] * np.exp(1j * phase)
        est = librosa.istft(
            complex_spec,
            hop_length=cfg.data.hop_length,
            win_length=cfg.data.n_fft
        )
        est_wavs.append(est)

    return est_wavs, sr

def load_references(track_dir, cfg):
    refs = []
    for src in cfg.data.sources:
        if src == "mixture":
            continue
        path = os.path.join(track_dir, f"{src}.wav")
        wav, _ = librosa.load(path, sr=cfg.data.sample_rate, mono=True)
        refs.append(wav)
    return refs

def evaluate_all(checkpoint, cfg, output_csv, output_dir):
    model, device = load_model(checkpoint, cfg)
    test_root = os.path.join(cfg.data.raw_path, "test")
    results = []

    for track_id in sorted(os.listdir(test_root)):
        track_dir = os.path.join(test_root, track_id)
        mix_wav   = os.path.join(track_dir, "mixture.wav")
        if not os.path.isfile(mix_wav):
            continue

        # --- 1) Separate & save audio ---
        est_wavs, sr = separate_track(mix_wav, model, device, cfg)
        out_dir = os.path.join(output_dir, track_id)
        os.makedirs(out_dir, exist_ok=True)
        for src_name, est in zip(cfg.data.sources[1:], est_wavs):
            out_path = os.path.join(out_dir, f"{src_name}.wav")
            sf.write(out_path, est, sr)
        print(f"Saved separated stems for track '{track_id}' → {out_dir}")

        # --- 2) Load ground-truth and compute metrics ---
        ref_wavs = load_references(track_dir, cfg)
        min_len = min(min(len(r) for r in ref_wavs), min(len(e) for e in est_wavs))
        ref_arr = np.vstack([r[:min_len] for r in ref_wavs])
        est_arr = np.vstack([e[:min_len] for e in est_wavs])
        sdr, sir, sar, _ = bss_eval_sources(ref_arr, est_arr)

        results.append({
            "track": track_id,
            "avg_sdr": np.mean(sdr),
            "avg_sir": np.mean(sir),
            "avg_sar": np.mean(sar),
        })
        print(f"Track {track_id}: SDR={sdr.mean():.2f}, SIR={sir.mean():.2f}, SAR={sar.mean():.2f}")

    # --- 3) Save CSV report ---
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\n=== Summary ===")
    print(df.describe().loc[["mean","std"]])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint of your UNet")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output", default="results/eval.csv",
                   help="CSV file to write per-track metrics")
    p.add_argument("--out-dir", default="results/separated",
                   help="Directory to write separated .wav files")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate_all(args.checkpoint, cfg, args.output, args.out_dir)
