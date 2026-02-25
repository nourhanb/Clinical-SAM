# Clinical-SAM

**Agentic, one-bit physician-in-the-loop prompting for clinical 3D CT segmentation with a frozen SAM.**

---

## Abstract

SAM-style foundation models can segment from prompts, but in clinical 3D CT they remain brittle: small prompt errors trigger large boundary mistakes, forcing repeated corrective interactions before a mask is clinically acceptable. We propose **Clinical-SAM**, a two-stage, agentic framework that improves the clinical acceptability of a frozen SAM while requiring only a **binary judgment** (accept/reject)—no corrective clicks.

- **Phase I (pre-deployment):** Train a self-prompting RL policy with dense ground-truth supervision to learn a robust geometric prompting prior, without consuming physician time.
- **Phase II (deployment):** Refine the policy online from terminal accept/reject feedback under a prequential protocol, shaping a latent acceptability prior and trading off mask quality against interaction cost.

Across public 3D CT tumor benchmarks, Clinical-SAM achieves strong Dice and boundary-sensitive NSD versus automated, interactive, and online-learning baselines, with consistently cleaner contours under minimal physician burden.

---

## Overview

This repository provides:

- **Phase I:** PPO-based training of a prompting policy (implicit agent) in a custom Gymnasium environment with RepViT-SAM, using dense reward signals from ground-truth masks.
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
venv/bin/python train_sam_align_ppo.py \
  --agent_cfg_path configs/agents/implicit_agent.yaml \
  --env_cfg_path configs/envs/repvit_sam_medical_improved.yaml \
  --num_envs 16 \
  --total_timesteps 5000000 \
  --checkpoint_iter_freq 100
```

Optional: run in the background and log stdout/stderr:

```bash
nohup venv/bin/python train_sam_align_ppo.py \
  --agent_cfg_path configs/agents/implicit_agent.yaml \
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

Override the Phase I checkpoint or output directory if needed (edit the YAML or add CLI options). Results and the adapted agent are written under `phase2/results_online/` (CSVs and `phase2_online_agent.pth`).

Background run with logging:

```bash
nohup venv/bin/python phase2_ppo_online.py \
  --config phase2/configs/phase2_config_online.yaml \
  > phase2/results_online/phase2_online_run.log 2>&1 &
tail -f phase2/results_online/phase2_online_run.log
```

### Evaluation (Phase I or Phase II agent)

Evaluate a trained agent (e.g. Phase I checkpoint or Phase II saved agent) on the **training** split:

```bash
venv/bin/python evaluate.py \
  --checkpoint_path weights/latest.pth \
  --agent_cfg_path configs/agents/implicit_agent.yaml \
  --env_cfg_path configs/envs/repvit_sam_medical_improved.yaml \
  --num_episodes 89
```

Evaluate on the **test** split:

```bash
venv/bin/python evaluate.py \
  --checkpoint_path weights/latest.pth \
  --agent_cfg_path configs/agents/implicit_agent.yaml \
  --env_cfg_path configs/envs/repvit_sam_medical_test.yaml \
  --num_episodes 26
```

Use `phase2/results_online/phase2_online_agent.pth` as `--checkpoint_path` to evaluate the Phase II–adapted policy (the eval script expects the same agent interface and state_dict).

---

## Configuration Summary

| Stage    | Config / Script | Main output |
|----------|------------------|-------------|
| Phase I  | `train_sam_align_ppo.py` + `configs/agents/implicit_agent.yaml`, `configs/envs/repvit_sam_medical_improved.yaml` | `weights/latest.pth` (or `logs/.../checkpoints/`) |
| Phase II | `phase2_ppo_online.py` + `phase2/configs/phase2_config_online.yaml` | `phase2/results_online/phase2_online_agent.pth`, CSVs |
| Eval     | `evaluate.py` + env/agent configs | Console metrics (Dice, IoU, etc.) |

---

## Checkpoints

Pre-trained checkpoints (if available) can be downloaded from the project’s [Hugging Face Space](https://huggingface.co/shantanusingh10/AlignSAM-unofficial). Place them in `weights/` and set `phase1_checkpoint` in Phase II config or use `--checkpoint_path` in evaluation accordingly.

---

## Project Structure (high level)

- `train_sam_align_ppo.py` — Phase I PPO training.
- `phase2_ppo_online.py` — Phase II online PPO (prequential, binary feedback).
- `evaluate.py` — Evaluation script for any saved agent.
- `models/` — Agent definitions (e.g. implicit agent used in both phases).
- `custom_gym_implns/` — Gymnasium environment (RepViT-SAM, medical dataset).
- `phase2/configs/` — Phase II YAML configs; `phase2/utils/` — GAE, Dice, KL, episode runner.

---

