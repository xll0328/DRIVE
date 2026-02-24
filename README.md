# DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.10330-b31b1b.svg)](https://arxiv.org/abs/2409.10330)
[![Project Page](https://img.shields.io/badge/Project%20Page-xll0328.github.io%2Fdrive-2563eb)](https://xll0328.github.io/drive/)

**This paper was accepted at ICRA 2025** (IEEE International Conference on Robotics and Automation 2025, CCF B, Core A\*). This repository is the official PyTorch Lightning implementation.

| [**Paper (PDF)**](https://arxiv.org/abs/2409.10330) | [**Project Page**](https://xll0328.github.io/drive/) | [**Code**](https://github.com/xll0328/ICRA25-DRIVE) |
|:---:|:---:|:---:|
| [arXiv](https://arxiv.org/abs/2409.10330) | [xll0328.github.io/drive](https://xll0328.github.io/drive/) | [GitHub](https://github.com/xll0328/ICRA25-DRIVE) |

**Authors:** Songning Lai~, Tianlang Xue~, Hongru Xiao, Lijie Hu, Jiemin Wu, Ninghui Feng, Runwei Guan, Haicheng Liao, Zhenning Li, Yutao Yue†

---

## Overview

**DRIVE** is a comprehensive framework for improving the **dependability and stability** of explanations in end-to-end unsupervised autonomous driving models. It addresses instability issues in concept-based explainable systems (e.g. built on [Concept Gridlock (DCG)](https://github.com/jessicamecht/concept_gridlock)) and provides both consistent/stable interpretability and consistent/stable control output.

- **Consistent interpretability** — explanations align with model behavior across samples.
- **Stable interpretability** — explanations remain robust under input perturbations.
- **Consistent output** — predictions are reliable across scenarios.
- **Stable output** — performance is robust under perturbations.

**Teaser:** see the [project page](https://xll0328.github.io/drive/) for the animated overview. Framework figure:

<p align="center">
  <img src="https://raw.githubusercontent.com/xll0328/xll0328_ed.github.io/main/project/DRIVE/framwork_all.png" width="90%" alt="DRIVE framework" />
</p>

---

## Table of Contents

- [Method](#method)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Arguments](#arguments)
- [Training & evaluation](#training--evaluation)
- [Analysis & scripts](#analysis--scripts)
- [Citation](#citation)
- [License](#license)

---

## Method

DRIVE defines four key attributes and implements a training/evaluation pipeline to achieve them:

1. **Consistent interpretability** — concept explanations are aligned with the model’s actual reasoning.
2. **Stable interpretability** — concept explanations remain stable under input noise/perturbations.
3. **Consistent output** — driving decisions are consistent across similar inputs.
4. **Stable output** — driving performance is robust under perturbations.

The codebase builds on a concept-bottleneck driving model: first train a DCG-style checkpoint, then resume with the DRIVE pipeline (see [Training & evaluation](#training--evaluation)) to optimize for the above properties. Novel metrics for evaluating dependability are included in the analysis notebooks.

---

## Installation

```bash
git clone https://github.com/xll0328/ICRA25-DRIVE.git
cd ICRA25-DRIVE
conda create -n drive python=3.8
conda activate drive
pip install -r DRIVE/requirement.txt
```

---

## Project structure

```
ICRA25-DRIVE/
├── README.md
├── LICENSE
├── DRIVE/
│   ├── requirement.txt
│   ├── dataloader_comma.py
│   ├── comma_preprocess/          # Data preprocessing (raw_readers, clean_up_data)
│   ├── DCG_Core/                  # Core DCG model and training
│   │   ├── main.py, model.py, module.py
│   │   └── README.md
│   └── DRIVE/
│       ├── main_copy1.py          # Main entry for DRIVE training/testing
│       ├── model_copy1.py, model_copy2.py, module_copy1.py
│       ├── losses.py, PGD.py
│       ├── sh_scripts/            # Training and evaluation scripts
│       ├── attacks/               # Perturbation and noise tools
│       ├── scenarios_DCG/         # Scenario configs
│       └── analysis/              # Notebooks for MAE, concept visualization, performance
```

---

## Setup

### 1. Dataset

- **Comma2k19:** [commaai/comma2k19](https://github.com/commaai/comma2k19)
- **Download:** [Academic Torrents](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb)

### 2. Data preprocessing

Extract video frames, obtain sample data, and save to HDF5:

- Run `DRIVE/comma_preprocess/raw_readers.ipynb`
- Run `DRIVE/comma_preprocess/clean_up_data.ipynb`

Paths inside the notebooks and scripts (e.g. dataset path, checkpoint path) should be set to your local environment.

---

## Arguments

| Argument | Example | Description |
|----------|---------|-------------|
| `-task` | `angle` / `distance` / `multitask` | Task for the model |
| `-train` | `True` | Enable training |
| `-test` | `False` | Enable testing |
| `-gpu_num` | `1` | Number of GPUs |
| `-dataset_path` | `/path/to/dataset` | Path to dataset and checkpoints |
| `-concept_features` | `True` | Use concept bottleneck |
| `-new_version` | `True` | Train from scratch (do not resume) |
| `-save_path` | `/path/to/savepath` | Path for saving CSV predictions |
| `-num-gpus` | `1` | Number of GPUs |
| `-max_epochs` | `200` | Maximum epochs |
| `-bs` | `4` | Batch size |
| `-checkpoint_path` | `path/to/checkpoint.ckpt` | Checkpoint for resume / test |
| `img_noise` | `GaussianNoise` | Noise type (build noisy dataset via `DRIVE/DRIVE/attacks/` if needed) |

---

## Training & evaluation

### 1. Train a DCG checkpoint (without DRIVE)

Follow [Concept Gridlock (DCG)](https://github.com/jessicamecht/concept_gridlock/blob/master/README.md) to obtain a DCG checkpoint.

### 2. Resume with DRIVE

From the repository root (or from `DRIVE/DRIVE/` if your paths are set there):

```bash
python main_copy1.py -dataset comma -backbone none -concept_features -ground_truth normal \
  -train -gpu_num 1 -max_epochs 50 -task distance -bs 2 -checkpoint_path DCG_checkpoint
```

More examples: see `DRIVE/DRIVE/sh_scripts/` (e.g. `run_experiments_concept_comma_DRIVE.sh`, `testconcept.sh`).

---

## Analysis & scripts

- **MAE and predictions:** `DRIVE/DRIVE/mae_test.ipynb` — load checkpoints, run prediction, compute MAE and save results.
- **Concept visualization:** `DRIVE/DRIVE/analysis/showconcepts.ipynb` — visualize concept activations and compare across conditions.
- **Model performance:** `DRIVE/DRIVE/analysis/observe_model_performances.ipynb` — aggregate metrics and plots.
- **Perturbation / noise:** `DRIVE/DRIVE/attacks/` — e.g. `noise-img.py` for adding Gaussian noise to HDF5 image data; adjust `h5_file_path` and output paths for your setup.

---

## Citation

```bibtex
@inproceedings{lai2025drive,
  title={DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving},
  author={Songning Lai and Tianlang Xue and Hongru Xiao and Lijie Hu and Jiemin Wu and Ninghui Feng and Runwei Guan and Haicheng Liao and Zhenning Li and Yutao Yue},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  eprint={2409.10330},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.10330},
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

## References

- [Concept Gridlock (DCG)](https://github.com/jessicamecht/concept_gridlock)
- [Comma2k19](https://github.com/commaai/comma2k19)
