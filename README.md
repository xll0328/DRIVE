# DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.10330-b31b1b.svg)](https://arxiv.org/abs/2409.10330)

**ICRA 2025 (CCF B, Core A\*)** · Official PyTorch Lightning implementation.

| [**Paper (PDF)**](https://arxiv.org/abs/2409.10330) | [**Project Page**](https://xll0328.github.io/drive/) | [**Code**](https://github.com/xll0328/ICRA25-DRIVE) |
|:---:|:---:|:---:|
| [arXiv](https://arxiv.org/abs/2409.10330) | [xll0328.github.io/drive](https://xll0328.github.io/drive/) | [GitHub](https://github.com/xll0328/ICRA25-DRIVE) |

---

## Overview

**DRIVE** is a comprehensive framework designed to address the reliability and stability of explanations in end-to-end unsupervised autonomous driving models (e.g. building on [DCG](https://github.com/jessicamecht/concept_gridlock)), improving dependability and robustness of such systems.

- **Consistent and stable interpretability** — explanations remain transparent and predictable across scenarios.
- **Consistent and stable output** — improved performance under perturbations and higher overall reliability.

**Framework overview:** see [Project Page](https://xll0328.github.io/drive/) or the figure below.

<p align="center">
  <img src="https://raw.githubusercontent.com/xll0328/xll0328_ed.github.io/main/project/DRIVE/framwork_all.png" width="90%" alt="DRIVE framework" />
</p>

---

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Arguments](#arguments)
- [Training & evaluation](#training--evaluation)
- [Citation](#citation)
- [License](#license)

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

## Setup

### 1. Dataset

- **Comma2k19:** [commaai/comma2k19](https://github.com/commaai/comma2k19)  
- **Download:** [Academic Torrents](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb)

### 2. Data preprocessing

Extract video frames, obtain sample data, and save to HDF5:

- Run `DRIVE/comma_preprocess/raw_readers.ipynb`
- Run `DRIVE/comma_preprocess/clean_up_data.ipynb`

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

From repo root (or `DRIVE/DRIVE/` as in the original scripts):

```bash
python main_copy1.py -dataset comma -backbone none -concept_features -ground_truth normal \
  -train -gpu_num 1 -max_epochs 50 -task distance -bs 2 -checkpoint_path DCG_checkpoint
```

More examples: see `DRIVE/DRIVE/sh_scripts/` (e.g. `run_experiments_concept_comma_DRIVE.sh`).

---

## Citation

```bibtex
@misc{lai2024drivedependablerobustinterpretable,
  title={DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving},
  author={Songning Lai and Tianlang Xue and Hongru Xiao and Lijie Hu and Jiemin Wu and Ninghui Feng and Runwei Guan and Haicheng Liao and Zhenning Li and Yutao Yue},
  year={2024},
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
