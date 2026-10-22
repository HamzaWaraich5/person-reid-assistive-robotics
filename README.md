# Robust Person Re-Identification for Assistive Robots

**Status: Preprint / research report — not peer reviewed.**

A single-run, single-dataset exploratory study of Person Re-ID robustness
under synthetic deployment-motivated degradations and adversarial
perturbations, including a matched-training control that isolates the
effect of adversarial fine-tuning from additional optimization alone.

Built on Market-1501 | ResNet-50 + BNNeck | PyTorch

**Full paper (PDF):** (https://doi.org/10.5281/zenodo.23537461)

---

## The Problem

Person Re-ID models score well on clean benchmarks, but a robot's camera
rarely produces clean images. This study measures how much accuracy is
lost under blur, occlusion, low brightness, and adversarial attack, and
tests whether adversarial fine-tuning or lightweight preprocessing
actually help — using a matched clean-training control to check whether
apparent gains are real or just the effect of extra training time.

## Key Finding

The matched control model — trained for the same extra epochs as the
adversarially fine-tuned model, but on clean data only — matches or
exceeds adversarial training on clean accuracy and on every tested
non-adversarial degradation. Adversarial training's one measurable
advantage is at stronger FGSM attack strengths.

| Condition | Baseline | Adv. Fine-Tuned | Clean-Training Control |
|---|---|---|---|
| Clean | 74.05% | 77.55% | **78.24%** |
| Blur (L2) | 1.63% | 3.71% | **3.83%** |
| Occlusion (L2) | 13.00% | 26.37% | **31.95%** |
| Brightness (L2) | 25.89% | 7.36% | **27.91%** |
| Perspective | 67.61% | 66.83% | **72.95%** |
| FGSM ε=0.01 | 73.40% | **77.11%** | 77.49% |
| FGSM ε=0.02 | 69.74% | **74.17%** | 72.98% |
| FGSM ε=0.05 | 45.78% | **53.09%** | 48.87% |

(Rank-1 accuracy, %. Bold = highest observed value per row. Full mAP
results in the paper.)

**Important caveat:** every number above is from a single training run
per configuration. The paper's Limitations section discusses this and
other scope restrictions in detail — read it before citing specific
numbers.

## Preprocessing Defenses

Lightweight, condition-specific preprocessing was tested separately on
clean-degraded images and on FGSM-attacked images, using a reconciled
evaluation pipeline:

| Operation | Targets | Clean-image cost | Added cost under FGSM attack |
|---|---|---|---|
| Sharpening | Blur | −2.28 pts | additional −6.33 pts |
| Upper-body crop | Occlusion | −51.48 pts | additional −1.43 pts |
| Histogram equalization | Low brightness | −19.41 pts | additional −6.69 pts |

Two of three operations show a larger loss under attack than their
clean-image cost alone predicts — a descriptive observation, not a
statistically established effect (single run, no variance estimate).
All operations add under 4ms on a reference CPU. Full breakdown and
caveats in the paper.

## Project Structure
person-reid-assistive-robotics/
│
├── paper/
│ └── deployment_robustness_study.pdf
│
├── notebooks/
│ ├── reid_training.ipynb ← baseline + matched-control training
│ └── reid_robustness_eval.ipynb ← degradation, FGSM, preprocessing experiments
│
├── results/
│ ├── threeway_comparison.png ← baseline / adv. FT / control chart
│ └── all_results.json ← complete numerical results
│
└── README.md

## Model

- Backbone: ResNet-50, pretrained on ImageNet
- Neck: BNNeck (BatchNorm1d, shift parameter frozen)
- Loss: CrossEntropyLoss + triplet loss (batch-hard mining)
- Input size: 256×128
- Optimizer: Adam, lr=3e-4 (baseline), lr=3e-5 (fine-tuning)

## Reproducibility

- Checkpoints: `reid_model_v4_bnneck.pth` (baseline), `reid_model_v5_pgd.pth`
  (adversarially fine-tuned, 5 epochs from v4), `reid_model_v4_control_nopgd.pth`
  (matched clean control, 5 epochs from v4)
- PGD: ε=0.02, α=0.005, 3 steps, normalized-space clipping [-3, 3]
- FGSM: single-step, same loss as PGD, evaluated at ε = 0.01 / 0.02 / 0.05
- Random seed: not fixed per run — single-seed study, see paper Limitations
- Transforms: resize 256×128, ImageNet normalization; train-time random
  flip, crop, color jitter
- Evaluation: Rank-1 and mAP, standard Market-1501 protocol
- Dataset: Market-1501, downloadable via Kaggle

## How to Run

```bash
git clone https://github.com/HamzaWaraich5/person-reid-assistive-robotics
```

Download Market-1501 from Kaggle, then open the notebooks in Google
Colab (a free T4 GPU is sufficient). `reid_training.ipynb` reproduces
the baseline and both fine-tuned models; `reid_robustness_eval.ipynb`
reproduces all degradation, attack, and preprocessing experiments.

## Degradation and Attack Protocol

All conditions are applied to Market-1501 query images only; the
gallery remains clean throughout.

- **Blur** — Gaussian filter, radius 1 / 3 / 5
- **Occlusion** — bottom-region black mask, 15% / 30% / 50%
- **Brightness** — pixel scaling, factor 0.6 / 0.3 / 0.1
- **Perspective** — horizontal perspective skew, one fixed setting
- **FGSM** — single-step gradient sign attack targeting embedding
  magnitude (not a validated metric-aware Re-ID attack — see paper
  Section III-C and Limitations)

## Scope and Limitations

This is a synthetic, single-dataset, single-backbone, single-seed
exploratory study. It does **not** evaluate a physical robot, embedded
hardware, or real camera data. The adversarial objective perturbs
feature magnitude rather than directly optimizing a retrieval-relevant
metric. Full discussion in the paper's Limitations section — please
read it before drawing conclusions beyond what's stated there.

## References

- Zheng et al., "Scalable Person Re-identification: A Benchmark," ICCV 2015
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
- Luo et al., "Bag of Tricks for Person Re-identification," CVPRW 2019
- Hermans et al., "In Defense of the Triplet Loss for Person Re-ID," arXiv 2017
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples," ICLR 2015
- Bai et al., "Adversarial Metric Attack and Defense for Person Re-ID," TPAMI 2021
- Wang et al., "Transferable, Controllable Adversarial Attacks on Person Re-ID," CVPR 2020
- Chen et al., "Benchmarks for Corruption Invariant Person Re-identification," NeurIPS 2021
- Zhuo et al., "Occluded Person Re-identification," ICME 2018
- Rollo et al., "CARPE-ID: Continuously Adaptable Re-ID for Personalized Robot Assistance," ARSO 2023
- Bardaro and Matteucci, "A Robust Monocular Person-Following System on Embedded Platforms," ECMR 2025

## License

[Choose one — MIT is a common default for code repos, and pairs fine
with the CC-BY 4.0 license on the Zenodo paper record]
