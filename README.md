# RAANet: Radial Attenuation Attention for Image Classification

Official PyTorch implementation of the paper *"Radial Attenuation Attention for Image Classification"* (The Journal of Supercomputing, 2026).

**Contact:** 18840950038@163.com

---

## Overview

RAANet introduces a biologically inspired attention mechanism that mimics the **center‑peripheral attenuation** property of the visual system. The core **RAA module** consists of:

- **RADConv** – directional convolutions with distance‑based attenuation (Ant function), enhancing central regions while suppressing peripheral background.
- **FAM** – channel‑wise feature modulation using the same Ant function to balance channel responses and suppress outliers.

The RAA module is embedded into ResNet‑34 residual blocks, consistently improving classification accuracy on CIFAR‑10/100, SVHN, Imagenette, Imagewoof, and ImageNet‑1K.

---



## Requirements

- Python 3.8+
- PyTorch ≥ 1.10 (recommended 2.6.0)
- torchvision, numpy, matplotlib

Install:
```bash
pip install torch torchvision numpy matplotlib

## Training
All hyperparameters follow the paper (SGD, lr=0.1, momentum=0.9, weight decay=5e‑4, 200 epochs, multi‑step decay at epochs 60/120/160, label smoothing 0.1).
