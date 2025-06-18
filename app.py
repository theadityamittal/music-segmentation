# serve.py

import os
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr
from omegaconf import OmegaConf

from src.models.unet import UNet

# 1) Load your config and model once at startup
CFG = OmegaConf.load("config/default.yaml")
DEVICE = torch.device("mps" if torch.mps else "cpu")

MODEL = UNet(
    in_ch=1,
    num_sources=len(CFG.data.sources) - 1,
    chans=CFG.model.chans,
    num_pool_layers=CFG.model.num_pool_layers
).to(DEVICE)

# point this at your best checkpoint in the Space
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "models/unet_best.pt")
MODEL.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
MODEL.eval()


def separate_file(mix_path):
    """
    Given a file path to the uploaded mixture WAV, returns
    a dict of { "drums": path, "bass": path, ... } to the separated .wav files.
    """
    # 1. Load audio & STFT
    wav, sr = librosa.load(mix_path, sr=CFG.data.sample_rate, mono=True)
    stft = librosa.stft(
        wav, n_fft=CFG.data.n_fft, hop_length=CFG.data.hop_length
    )
    mag, phase = np.abs(stft), np.angle(stft)
    F, T = mag.shape

    # 2. Pad to multiple of segment_length
    SEG = CFG.data.segment_length
    pad = (SEG - (T % SEG)) % SEG
    if pad:
        mag   = np.pad(mag,   ((0,0),(0,pad)), constant_values=0)
        phase = np.pad(phase, ((0,0),(0,pad)), constant_values=0)
    n_seg = mag.shape[1] // SEG

    # 3. Inference in chunks
    preds = []
    with torch.no_grad():
        for i in range(n_seg):
            mseg = mag[:, i*SEG:(i+1)*SEG]
            x = torch.from_numpy(mseg).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            y = MODEL(x)  # (1, S, F, SEG)
            preds.append(y.squeeze(0).cpu().numpy())
    pred_mag = np.concatenate(preds, axis=2)[:, :, :T]
    phase    = phase[:, :T]

    # 4. Reconstruct waveforms and write to temp files
    out_paths = {}
    for idx, src in enumerate(CFG.data.sources[1:]):
        spec = pred_mag[idx] * np.exp(1j * phase)
        est  = librosa.istft(
            spec,
            hop_length=CFG.data.hop_length,
            win_length=CFG.data.n_fft
        )
        # write to a temp WAV file
        fd, path = tempfile.mkstemp(suffix=f"_{src}.wav")
        os.close(fd)
        sf.write(path, est, sr)
        out_paths[src] = path

    # return in the order drums, bass, other, vocals
    return [out_paths[src] for src in CFG.data.sources[1:]]


# 5) Build Gradio interface
description = """
## Music Source Separation

Upload a mix `.wav` and get back **drums**, **bass**, **other**, and **vocals** stems separated by a U-Net model.
"""

iface = gr.Interface(
    fn=separate_file,
    inputs=gr.Audio(label="Mixture (.wav)", type="filepath"),
    outputs=[
         gr.Audio(label="Drums",  type="filepath"),
         gr.Audio(label="Bass",   type="filepath"),
         gr.Audio(label="Other",  type="filepath"),
         gr.Audio(label="Vocals", type="filepath"),
     ],
    title="U-Net Music Separator",
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=True)
