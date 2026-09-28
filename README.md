# HyperFuse

HyperFuse: Lightweight Flexible Printed Circuit Defect-Sample Synthesis via Similarity-Guided Gradient Fusion in Hyperbolic Space

## Introduction

This paper proposes HyperFuse, a lightweight local defect-sample synthesis framework for FPC AOI. It combines NCC patch retrieval, Poisson blending, and Poincaré-ball quality screening to generate high-fidelity FPC defect samples without training generative models.
## Components

python == 3.9
torch == 2.1.2
torchvision == 0.16.2
ultralytics == 8.2.0
opencv-python == 4.9.0.80
numpy == 1.24.3
diffusers == 0.27.2
lpips
pytorch-fid

## Use

First deploy the project, configure the environment, and run the scripts in order.

python split_dataset.py --src_img <imgs> --src_lbl <labels> --out dataset_baseline
python gen_candidate_pool.py --real_img <train_imgs> --real_lbl <train_labels> --out dataset_10k
python poincare_filter.py --real_img <train_imgs> --real_lbl <train_labels> --pool_img dataset_10k/images/train --pool_lbl dataset_10k/labels/train --out_img dataset_ours/images/train --out_lbl dataset_ours/labels/train --top_k 5000
python train_yolo.py --train dataset_baseline/images/train,dataset_ours/images/train --val dataset_baseline/images/val --name ours
python eval_detection.py --weights runs/detect/ours/best.pt --img_dir <test_imgs> --lbl_dir <test_labels> --name eval_ours

## How to see the results?

You can run the program to visualize the training curves and detection metrics (mAP@0.5, recall, precision) after the end of the program.
