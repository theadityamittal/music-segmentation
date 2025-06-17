import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.transforms      import SpectrogramTransforms
from src.data.dataset        import AudioDataset
from src.models.unet         import UNet

log = __import__('logging').getLogger(__name__)

@hydra.main(config_path="../config", config_name="default")
def main(cfg: DictConfig):
    # 1. Config dump
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Datasets & DataLoaders
    train_tx = SpectrogramTransforms(**cfg.augment)
    val_tx   = None  # no augmentation for validation

    train_ds = AudioDataset(split="train", cfg=cfg, transform=train_tx)
    val_ds   = AudioDataset(split="test", cfg=cfg, transform=val_tx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # 3. Model, optimizer, loss
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_ch=1,
        num_sources=len(cfg.data.sources)-1,  # exclude mixture
        chans=cfg.model.chans,
        num_pool_layers=cfg.model.num_pool_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.L1Loss()

    # 4. MLflow setup
    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run(run_name=cfg.experiment.run_name):
        # log hyperparams
        mlflow.log_params({
            "lr": cfg.training.lr,
            "batch_size": cfg.data.batch_size,
            "chans": cfg.model.chans,
            "num_pools": cfg.model.num_pool_layers
        })

        best_val_loss = float("inf")
        for epoch in range(1, cfg.training.epochs + 1):
            # --- Training ---
            model.train()
            train_loss = 0.0
            for step, (mix, target) in enumerate(train_loader, 1):
                mix, target = mix.to(device), target.to(device)
                # add channel dim to mix: (B,1,F,T)
                mix = mix.unsqueeze(1)
                pred = model(mix)  # (B, S, F, T)
                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * mix.size(0)
                # early exit for smoke test
                if cfg.training.max_steps is not None and step >= cfg.training.max_steps:
                    break

            train_loss /= len(train_ds)
            log.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for step, (mix, target) in enumerate(val_loader, 1):
                    mix, target = mix.to(device), target.to(device)
                    mix = mix.unsqueeze(1)
                    pred = model(mix)
                    val_loss += criterion(pred, target).item() * mix.size(0)
                    if cfg.training.max_steps is not None and step >= cfg.training.max_steps:
                        break
            val_loss /= len(val_ds)
            log.info(f"[Epoch {epoch}]  Val Loss: {val_loss:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(
                    hydra.utils.get_original_cwd(),
                    "models",
                    f"unet_epoch{epoch}_val{val_loss:.4f}.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(ckpt_path)
                log.info(f"Saved checkpoint: {ckpt_path}")

    log.info("Training complete.")

if __name__ == "__main__":
    main()
