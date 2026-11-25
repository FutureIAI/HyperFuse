# HyperFuse: Training-free Hyperbolic-space Fusion for PCB/FPC Defect Augmentation

Official code for the paper *"Training-free HyperFuse: Similarity-Guided Fusion for Small-Defect Detection on FPC/PCB"*.

## Overview

HyperFuse is a **training-free** augmentation pipeline that synthesizes defect samples for small-defect detection without training a task-specific generative model. It has three stages:

1. **SCCP (Spectral-Correlation-Constrained Placement)** — global search over training backgrounds, with hard gates on trace direction and periodicity, and a soft composite score (gradient / trace / phase).
2. **Poisson blending** — seamless boundary blending of the bbox defect patch onto the target background.
3. **Poincaré-ball quality screening** — ResNet-18 features mapped into the Poincaré ball (curvature c=1), Frechet-mean anchor, 95%-quantile distance threshold, keep top-K candidates.

## Repository layout

```
HyperFuse_release/
├── src/
│   ├── sccp_search.py          # Stage 1: global placement search
│   ├── poisson_fusion.py       # Stage 2: Poisson blending + residual mask
│   ├── poincare_filter.py      # Stage 3: Poincaré screening
│   ├── residual_mask.py       # FPC defect residual extraction
│   ├── placement_features.py  # 8-region gradient / trace features
│   ├── trace_direction.py     # direction gate diagnostics
│   └── trace_available.py     # period/availability gate
├── baselines/
│   ├── cutpaste.py            # Cut-Paste baseline
│   ├── wgan_gp_train.py       # WGAN-GP training
│   ├── wgan_gp_generate.py    # WGAN-GP sampling
│   └── sd_inpaint.py          # SD-Inpaint baseline
├── scripts/
│   ├── split_fpc.py           # board-level 1514/300/400 split
│   ├── prepare_pku.py         # PKU-Market-PCB 80/20 split
│   ├── eval_quality.py        # FID / SSIM / LPIPS
│   ├── eval_detection.py      # COCO mAP
│   └── train_yolo.py          # YOLO11s / YOLOv8s training (seeds 42/2024/2025)
├── pipeline.py                # end-to-end entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset layout (YOLO format)

```
data/fpcsd/
├── images/
│   ├── train/   (1514)
│   ├── val/     (300)
│   └── test/    (400)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Board-level split (no same board across train/val/test).

## Usage

### Stage 1+2: SCCP placement + Poisson fusion

```bash
export HYPERFUSE_DATA_ROOT=./data/fpcsd
export HYPERFUSE_OUT=./outputs/full_run
python src/sccp_search.py
```

This enumerates candidate windows across all training backgrounds (excluding the source image itself), applies direction/periodicity hard gates, and writes Poisson-blended synthesized samples.

### Stage 3: Poincaré screening

```bash
python src/poincare_filter.py \
    --real_img   ./data/fpcsd/images/train \
    --real_lbl   ./data/fpcsd/labels/train \
    --pool_img   ./outputs/full_run/images \
    --pool_lbl   ./outputs/full_run/labels \
    --out_img    ./outputs/kept/images \
    --out_lbl    ./outputs/kept/labels \
    --top_k 5000 --quantile 0.95 --seed 42
```

### Training detectors

```bash
python scripts/train_yolo.py --model yolov8s --seed 42
python scripts/train_yolo.py --model yolov11s --seed 2024
```

## Key hyperparameters (paper defaults)

| Item | Value |
|---|---|
| Feature encoder | ResNet-18 (ImageNet pretrained, fc removed) |
| Feature dim | 512, L2 normalize × 0.99 |
| Poincaré curvature | c = 1 |
| Frechet mean iterations | 20, weighted by 1/d_H |
| Distance threshold | 95th percentile of real-sample distances |
| Candidate pool | 10,000 candidates → 5,000 kept |
| Seeds | 42 / 2024 / 2025 (mean ± std) |
| Placement weights | λ_trace=0.1, λ_period=0.05, λ_phase=0.05 |
| Safe margin | 15 px |

## Notes

- The FPCSD-EXP2024 dataset used in the paper is not publicly released due to commercial restrictions; code and split manifests are provided.
- PKU-Market-PCB is publicly available; our board-level 80/20 split is in `scripts/prepare_pku.py`.
- Peak GPU memory / per-sample timing numbers in Table 7 should be re-measured on your hardware; the numbers in the paper are reported on an RTX 3090 Ti with an 8-worker CPU pool.

## Citation

```bibtex
@misc{hyperfuse2026,
  title={Training-free HyperFuse: Similarity-Guided Fusion for Small-Defect Detection on FPC/PCB},
  year={2026}
}
```
