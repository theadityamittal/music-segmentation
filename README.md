# Music Source Separation with U-Net

A PyTorch implementation of a U-Net–based music source separator, taking stereo mixes → drums, bass, other & vocals. Includes a full data pipeline (DVC’ed), training, evaluation and a Gradio-powered inference server (for Hugging Face Spaces).

---
## 🎵 Music Source Separation Demo

This Gradio Space lets you upload any mono `.wav` music mix and instantly separate it into **drums**, **bass**, **other**, and **vocals** stems using a pretrained U-Net model.

👉 Try it live: https://huggingface.co/spaces/theadityamittal/music-separator-space

---

## 📂 Repository Structure


```
.
├── config/
│   └── default.yaml           # all hyperparameters & paths
├── data/
│   ├── raw/                   # DVC-tracked raw ZIPs & extracted WAVs
│   └── processed/             # DVC-tracked HDF5 spectrogram segments
├── models/
│   └── checkpoints/           # saved `.pt` weights from training
├── results/
│   ├── eval.csv               # per-track SDR/SIR/SAR results
│   └── separated/             # WAV stems produced by evaluation
├── src/
│   ├── data/
│   │   ├── loader.py          # DVC + ZIP extraction & `.mp4 → .wav` stems
│   │   ├── preprocess.py      # STFT & segmentation → `.pt` or HDF5
│   │   ├── preprocess\_utils.py# `load_audio`, `compute_stft`
│   │   ├── dataset.py         # `AudioDataset` loading HDF5, lazy per-worker
│   │   └── transforms.py      # spectrogram augmentations
│   ├── models/
│   │   └── unet.py            # U-Net architecture
│   ├── train.py               # training loop (Hydra + MLflow + checkpoints)
│   └── serve.py               # Gradio app for inference (Hugging Face Space)
├── dvc.yaml                   # DVC pipeline: raw ZIP → processed HDF5
├── evaluate.py                # script to compute SDR/SIR/SAR & save stems
├── requirements.txt           # Python deps
└── README.md                  # you are here

```

---

## 🚀 Quickstart

### 1. Install dependencies

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch data via DVC

```bash
dvc pull             
```

### 3. Reproduce locally

```bash
dvc repro
```

---

## 🛠️ Training

Train the U-Net on the processed HDF5 segments, log to MLflow, and checkpoint best-validation models.

```bash
python -m src.train \
  training.epochs=50 \
  data.batch_size=16 \
  data.num_workers=4 \
  model.chans=32 \
  model.num_pool_layers=4 \
  experiment.name=full_run \
  experiment.run_name=first_training
```

For a quick smoke test on 5 batches:

```bash
python -m src.train \
  training.epochs=1 \
  training.max_steps=5 \
  data.batch_size=2 \
  data.num_workers=0 \
  experiment.name=smoke_test \
  experiment.run_name=small_batch
```

Checkpoints are saved under `models/checkpoints/` and logged as MLflow artifacts.

---

## 📊 Evaluation

Compute BSS-Eval metrics and save separated WAV stems on the **test** split:

```bash
python -m src.evaluate \
  --checkpoint models/unet_best.pt \
  --config config/default.yaml \
  --output results/eval.csv \
  --out-dir results/separated
```

* **results/eval.csv**: per-track SDR/SIR/SAR
* **results/separated/**: ground-truth stems for listening

---

## 🎛️ Inference & Demo

Run the Gradio app locally:

```bash
python serve.py
```

Then open [http://localhost:7860](http://localhost:7860) to upload a `.wav` mix and download separated stems.

---

## ☁ Deployment on Hugging Face Spaces

1. Create a new Space, choose **Gradio** + **GPU**.
2. Push this repo (with `serve.py`, `config/`, `src/`, `requirements.txt`, `models/unet_best.pt`) to the Space’s Git.
3. In the Space settings, set `CHECKPOINT_PATH=models/unet_best.pt`.
4. The Space will auto-build and serve a web UI.

---

## 🔧 DVC Pipeline

```yaml
stages:
  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - config/default.yaml
      - data/raw/train.zip
      - data/raw/test.zip
    outs:
      - data/processed/train.h5
      - data/processed/test.h5
```

Reproduce everything with:

```bash
dvc repro        # preprocess → processed data
dvc push         # upload raw & processed artifacts
```

---

## 📈 Next Steps

* Improve model: complex-valued U-Net, STFT loss, phase reconstruction
* Hyperparameter sweep with Hydra + Optuna
* CI: smoke-train on PRs, linting & type checks
* Dockerize & add CI/CD for automated model serving

---

## 📜 License

[MIT License](./LICENSE)

Feel free to raise issues or contribute!
