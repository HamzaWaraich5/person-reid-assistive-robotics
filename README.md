# Robust Person Re-Identification for Assistive Robots

A systematic study of Re-ID robustness under real-world robot deployment 
conditions, with lightweight and adversarial training defenses.

Built on Market-1501 benchmark | ResNet50 + BN Neck | PyTorch

---

## The Problem

Person Re-Identification models achieve strong results in controlled 
benchmarks — but how do they perform under conditions an assistive 
robot actually faces?

This project answers that question and proposes a two-tier defense strategy.

---

## Key Findings

| Condition | Baseline | PGD Training | Two-Tier Defense |
|---|---|---|---|
| Clean | 74.05% | 77.55% | 77.55% |
| Blur | 1.63% | 3.92% | **7.45%** |
| Occlusion | 13.00% | **33.08%** | 26.22% |
| Low-light | 25.89% | 9.47% | **25.86%** |
| Viewpoint | 67.61% | 71.88% | 71.88% |
| FGSM ε=0.01 | 73.66% | **77.11%** | — |
| FGSM ε=0.02 | 69.06% | **74.17%** | — |
| FGSM ε=0.05 | 45.43% | **53.09%** | — |

**Key insight:** Real-world conditions are more dangerous than adversarial 
attacks for assistive robots. Motion blur alone drops accuracy from 
74% to 1.63% — near random guessing.

---

## Two-Tier Defense Strategy

| Defense | Targets | Latency overhead |
|---|---|---|
| PGD adversarial training | Occlusion, FGSM attacks | None at inference |
| Image sharpening | Blur | +0.51ms |
| Upper body crop | Occlusion | +0.01ms |
| Histogram equalization | Low-light | +3.53ms |

All defenses add less than 4ms overhead — well within real-time 
constraints for embedded robotics hardware (Jetson Xavier NX target: 
<100ms per frame).

---

## Results Chart

![Robustness Results](results/final_chart_professional.png)

---

## Project Structure

```
person-reid-assistive-robotics/
│
├── notebooks/
│   ├── reid_baseline.ipynb        ← baseline model training
│   └── reid_robustness.ipynb      ← robustness experiments
│
├── results/
│   ├── final_chart_professional.png
│   ├── fgsm_chart_professional.png
│   └── final_results_complete.json
│
└── README.md
```

---

## Model Architecture

- **Backbone:** ResNet50 pretrained on ImageNet
- **Neck:** BN Neck (Batch Normalization before classifier)
- **Loss:** CrossEntropyLoss + Triplet Loss
- **Defense:** PGD adversarial training (ε=0.02, 3 steps, 5 epochs)
- **Input size:** 256×128 (standard Re-ID format)

---

## Training Progress

| Version | Change | Rank-1 |
|---|---|---|
| v1 | Baseline ResNet50 | 61.22% |
| v2 | Augmentation + longer training | 67.40% |
| v3 | Triplet Loss | 71.17% |
| v4 | BN Neck | 74.05% |
| v5 | PGD adversarial training | 77.55% |

---

## How to Run

**1. Clone and setup**
```bash
git clone https://github.com/YOUR_USERNAME/person-reid-assistive-robotics
```

**2. Download Market-1501 dataset**
Available on Kaggle: search "Market-1501 dataset"

**3. Run in Google Colab**
- Open `notebooks/reid_baseline.ipynb` for training
- Open `notebooks/reid_robustness.ipynb` for robustness experiments
- GPU recommended (T4 on free Colab is sufficient)

---

## Robustness Conditions Tested

Conditions are simulated programmatically on Market-1501 query images:

- **Blur** — Gaussian blur radius 3 (robot camera motion)
- **Occlusion** — Bottom 30% blacked out (obstacle in path)
- **Low-light** — Brightness reduced to 30% (corridor at night)
- **Viewpoint** — Perspective transform (robot camera angle)
- **FGSM** — Fast Gradient Sign Method at ε=0.01, 0.02, 0.05

---

## Motivation

This work is inspired by research at the 
[NICR Lab, TU Ilmenau](https://www.tu-ilmenau.de/neurob) 
on assistive robotics and cognitive robot systems.

A robot that fails to recognize people under common real-world 
conditions is a safety problem — not just a benchmark limitation.

---

## References

- Zheng et al., "Scalable Person Re-identification: A Benchmark", ICCV 2015
- Luo et al., "Bag of Tricks for Person Re-identification", CVPR 2019
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
- Bai et al., "Adversarial Metric Attack and Defense for Person Re-identification", TPAMI 2021
