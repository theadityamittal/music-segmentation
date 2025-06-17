# src/data/transforms.py

import random
import torch

class SpectrogramTransforms:
    """
    Applies spectrogram augmentations:
      - time masking
      - frequency masking
      - time warping (circular shift)
      - stripe dropout (random thin zero-stripes)
      - additive noise
    """

    def __init__(self,
                 time_mask_param: int = 30,
                 freq_mask_param: int = 15,
                 time_warp_param: int = 40,
                 stripe_time_width: int = 1,
                 stripe_freq_width: int = 1,
                 stripe_time_count: int = 2,
                 stripe_freq_count: int = 2,
                 noise_std: float      = 0.01):
        self.time_mask_param   = time_mask_param
        self.freq_mask_param   = freq_mask_param
        self.time_warp_param   = time_warp_param
        self.stripe_time_width = stripe_time_width
        self.stripe_freq_width = stripe_freq_width
        self.stripe_time_count = stripe_time_count
        self.stripe_freq_count = stripe_freq_count
        self.noise_std         = noise_std

    def time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        _, T = spec.shape
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, max(0, T - t))
        spec[:, t0:t0 + t] = 0
        return spec

    def freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        F, _ = spec.shape
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, max(0, F - f))
        spec[f0:f0 + f, :] = 0
        return spec

    def time_warp(self, spec: torch.Tensor) -> torch.Tensor:
        # simple circular shift along time axis
        _, T = spec.shape
        w = random.randint(-self.time_warp_param, self.time_warp_param)
        return torch.roll(spec, shifts=w, dims=1)

    def stripe_dropout(self, spec: torch.Tensor) -> torch.Tensor:
        F, T = spec.shape
        # drop time stripes
        for _ in range(self.stripe_time_count):
            t0 = random.randint(0, max(0, T - self.stripe_time_width))
            spec[:, t0:t0 + self.stripe_time_width] = 0
        # drop freq stripes
        for _ in range(self.stripe_freq_count):
            f0 = random.randint(0, max(0, F - self.stripe_freq_width))
            spec[f0:f0 + self.stripe_freq_width, :] = 0
        return spec

    def add_noise(self, spec: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # Apply each augment with 50% chance
        if random.random() < 0.5:
            spec = self.time_warp(spec)
        if random.random() < 0.5:
            spec = self.time_mask(spec)
        if random.random() < 0.5:
            spec = self.freq_mask(spec)
        if random.random() < 0.5:
            spec = self.stripe_dropout(spec)
        if random.random() < 0.5:
            spec = self.add_noise(spec)
        return spec
