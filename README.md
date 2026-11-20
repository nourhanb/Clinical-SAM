# Clinical-SAM: Agentic Test-Time Adaptation of SAM via Physician-in-the-Loop for Clinical Tumor Segmentation

---

## Abstract

Interactive foundation models such as SAM struggle to generalize across clinical deployment sites. Among existing adaptation strategies, test-time adaptation (TTA) is the most promising, yet current TTA methods trade one problem for another: removing the physician eliminates clinical grounding, while retaining the physician reintroduces a costly annotation burden. In this work, we propose Clinical-SAM, a two-phase agentic physician-in-the-loop TTA framework that teaches a frozen SAM to approximate what a physician would accept - directly and continuously, without requiring spatial annotations. In Phase I (pre-deployment), a self-prompting agent trained via reinforcement learning generates prompts autonomously, eliminating per-case expert spatial input at inference. In Phase II (deployment), the agent policy is refined online by the one signal physicians already produce: a binary accept/reject feedback on each proposed mask. As the binary feedback accumulates across deployed cases, the agent converges toward a policy that produces masks meeting a reference acceptance standard designed to approximate physician judgment, with progressively fewer interactions. We evaluate Clinical-SAM in a pre-clinical setting on six oncological benchmarks spanning CT (colon, pancreas, liver, kidney) and PET (head & neck, lung) tumor segmentation, using agreement with reference-standard segmentations as a proxy for physician acceptance. Clinical-SAM achieves state-of-the-art performance on all six datasets, outperforming fine-tuning, interactive, TTA, and online-learning baselines, while requiring only 3.3 binary judgments per case on average across the six datasets. Our code is available here.

---

## Overview

This repository provides:

- **Phase I:** PPO-based training of a prompting policy in a custom Gymnasium environment with RepViT-SAM, using dense reward signals from ground-truth masks.
- **Phase II:** Online adaptation from simulated physician feedback (accept if Dice ≥ τ, else reject), with terminal rewards, a replay buffer, and KL regularization toward the Phase I policy.
- Medical 3D CT support (e.g., Medical Segmentation Decathlon), YAML configs, and evaluation scripts.

---

## Installation

**Requirements:** Python 3.8+, CUDA for GPU training.

```bash
git clone https://github.com/nourhanb/Clinical-SAM.git
cd Clinical-SAM
```

Create and activate a virtual environment, then install dependencies from the project root:

```bash
bash install_dependencies.sh
```

This installs the requirements, clones [RepViT](https://github.com/THU-MIG/RepViT) (the RepViT-SAM backbone), and downloads the RepViT-SAM weights to `RepViT/sam/weights/repvit_sam.pt`.

Place your medical dataset (e.g., Task10_Colon) under `images/` and provide train/val/test splits (e.g., `split.pkl`). See `configs/envs/repvit_sam_medical_*.yaml` for `data_dir`, `split_pkl_path`, and related options.

---

## How to Run

All commands below assume you are in the project root and use the project’s virtualenv (e.g. `venv/bin/python`). Set `PYTHONPATH` if your environment requires it: `export PYTHONPATH=$(pwd)`.

### Phase I — Train the prompting policy (pre-deployment)

Train the RL policy with PPO using dense ground-truth supervision. Checkpoints are written to the path specified in your run (e.g. `weights/latest.pth` or under `logs/`).

```bash
venv/bin/python train_sam_agent_ppo.py \
  --agent_cfg_path configs/agents/ppo_agent.yaml \
  --env_cfg_path configs/envs/repvit_sam_medical_improved.yaml \
  --num_envs 16 \
  --total_timesteps 5000000 \
  --checkpoint_iter_freq 100
```

Optional: run in the background and log stdout/stderr:

```bash
nohup venv/bin/python train_sam_agent_ppo.py \
  --agent_cfg_path configs/agents/ppo_agent.yaml \
  --env_cfg_path configs/envs/repvit_sam_medical_improved.yaml \
  --num_envs 16 \
  --total_timesteps 5000000 \
  --checkpoint_iter_freq 100 \
  > phase1_training.log 2>&1 &
```

### Phase II — Online adaptation from binary feedback (deployment)

After Phase I, run Phase II to adapt the policy from simulated physician feedback (accept/reject by Dice threshold). Config: `phase2/configs/phase2_config_online.yaml` (paper settings: τ_acc=0.75, K_max=11, B=3, buffer=30, η=0.02, KL weight=0.1).

```bash
venv/bin/python phase2_ppo_online.py \
  --config phase2/configs/phase2_config_online.yaml
```


---
## Checkpoints

Place the Phase I checkpoint in `weights/` and set `phase1_checkpoint` in the Phase II config (or use `--checkpoint_path` in evaluation) accordingly.

---

## Acknowledgements

This code builds on the MIT-licensed AlignSAM-CVPR2024-Unofficial implementation by Shantanu Singh (see `LICENSE`).

---

## Project Structure (high level)

To be added ... 
