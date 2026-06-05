# Clinical-SAM

**Agentic Test-Time Adaptation of SAM via Physician-in-the-Loop  for Clinical Tumor Segmentation.**

---

## Abstract

Interactive foundation models such as SAM struggle to generalize across clinical deployment sites. Among existing adaptation strategies, test-time adaptation (TTA) is the most promising, yet current TTA methods trade one problem for another: removing the physician eliminates clinical grounding, while retaining the physician reintroduces a costly annotation burden.  In this work, we propose Clinical-SAM, a two-phase agentic physician-in-the-loop TTA framework that teaches a frozen SAM what a  physician accepts - directly, continuously, and at zero annotation cost. In Phase I (pre-deployment), a self-prompting agent trained via reinforcement learning generates prompts autonomously, eliminating per-case expert input at inference. In Phase II (deployment), the agent policy is refined online by the one signal physicians already produce at zero marginal cost: a binary accept or reject on each proposed mask. As binary feedback accumulates across deployed cases, the agent learns the physician's acceptability standard, converging toward a policy that produces clinically acceptable masks with progressively fewer binary interactions. We evaluate Clinical-SAM on six oncological benchmarks spanning CT (colon, pancreas, liver, kidney) and PET (head & neck, lung) tumor segmentation. Clinical-SAM achieves state-of-the-art performance on all six datasets, outperforming fine-tuning, interactive, TTA, and online-learning baselines, while requiring a mean of only 3.3 binary judgments per case. 

- **Phase I (pre-deployment):** Train a self-prompting RL policy with dense ground-truth supervision to learn a robust geometric prompting prior, without consuming physician time.
- **Phase II (deployment):** Refine the policy online from terminal accept/reject feedback under a prequential protocol, shaping a latent acceptability prior and trading off mask quality against interaction cost.

Across six public PET and CT tumor benchmarks, Clinical-SAM achieves strong Dice and boundary-sensitive NSD versus automated, interactive, and online-learning baselines, with consistently cleaner contours under minimal physician burden.

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
git clone --recurse-submodules <repository-url>
cd Clinical-SAM
```

Create and activate a virtual environment, then install dependencies from the project root:

```bash
bash install_dependencies.sh
```

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

### Phase II — Online adaptation from binary feedback

After Phase I, run Phase II to adapt the policy from simulated physician feedback (accept/reject by Dice threshold). Config: `phase2/configs/phase2_config_online.yaml` (paper settings: τ_acc=0.75, K_max=11, B=3, buffer=30, η=0.02, KL weight=0.1).

```bash
venv/bin/python phase2_ppo_online.py \
  --config phase2/configs/phase2_config_online.yaml
```


---
## Checkpoints

Pre-trained checkpoints (if available) can be downloaded from the project’s [Hugging Face Space](https://huggingface.co/shantanusingh10/AlignSAM-unofficial). Place them in `weights/` and set `phase1_checkpoint` in Phase II config or use `--checkpoint_path` in evaluation accordingly.

---

## Project Structure (high level)

To be added ... 
