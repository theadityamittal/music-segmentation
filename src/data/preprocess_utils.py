import numpy as np
import torch
import librosa
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("config/default.yaml")

# Audio parameters from config
target_sr = cfg.data.sample_rate
n_fft = cfg.data.n_fft
hop_length = cfg.data.hop_length


def load_audio(path: str, sr: int = target_sr) -> np.ndarray:
    """
    Load a mono audio file at the target sample rate.

    Args:
        path: Path to the .wav file.
        sr: Desired sample rate.

    Returns:
        1D numpy array of audio samples.
    """
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def compute_stft(wav: np.ndarray) -> torch.Tensor:
    """
    Compute the complex STFT of an audio signal and return a tensor.

    Args:
        wav: 1D numpy array of audio samples.

    Returns:
        Tensor of shape (2, freq_bins, time_frames) with real and imaginary parts.
    """
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    real = np.real(stft)
    imag = np.imag(stft)
    spec = np.stack([real, imag], axis=0)
    return torch.from_numpy(spec).float()
