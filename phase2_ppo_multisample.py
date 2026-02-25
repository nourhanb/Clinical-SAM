"""
Phase II Multi-Sample PPO: Same prequential test-then-learn as online, but collects
multiple trajectories per test case for more stable updates.

Paper §2.3: Same reward and update rule; here we run trajectories_per_sample episodes
per case in each batch, then update from the combined buffer.
"""
import os
import sys
import csv
import copy
import argparse
from collections import deque

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import make_agent
import custom_gym_implns  # noqa: F401
from phase2.utils.adaptation_utils import (
    compute_dice_score,
    simulate_physician_feedback,
    compute_gae,
    compute_kl_divergence,
)
from phase2.utils.episode_runner import run_episode, load_obs_to_tensor

# Reuse helpers from online script
from phase2_ppo_online import (
    _get_ground_truth_from_info,
    compute_terminal_reward,
    trajectory_with_terminal_reward,
    stack_observations,
    flatten_trajectories_to_batch,
    OnlineBuffer,
    ppo_update_online,
)


def run_multisample_learning(config_path: str, output_dir: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if output_dir is None:
        output_dir = cfg.get("output_dir", "phase2/results_multisample")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    env_id = cfg.get("env_id", "SamSegEnv-v0")
    env_cfg_path = cfg.get("env_cfg_path")
    agent_cfg_path = cfg.get("agent_cfg_path")
    phase1_checkpoint = cfg.get("phase1_checkpoint")
    num_test_cases = cfg.get("num_test_cases", 26)
    seed = cfg.get("seed", 42)
    samples_per_batch = cfg.get("samples_per_batch", 5)
    trajectories_per_sample = cfg.get("trajectories_per_sample", 2)
    max_steps = cfg.get("max_adaptation_steps", cfg.get("max_steps", 50))
    acceptance_threshold = cfg.get("acceptance_threshold", 0.95)
    prompt_penalty = cfg.get("prompt_penalty", 0.02)
    buffer_size = cfg.get("buffer_size", 64)
    buffer_drop_easy_first = cfg.get("buffer_drop_easy_first", False)
    buffer_sample_size = cfg.get("buffer_sample_size")
    update_epochs = cfg.get("update_epochs", cfg.get("ppo", {}).get("update_epochs", 4))
    ppo_cfg = cfg.get("ppo", {})
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    save_results = cfg.get("save_results", True)
    verbose = cfg.get("verbose", True)

    env_cfg = yaml.safe_load(open(env_cfg_path, "r"))
    agent_cfg = yaml.safe_load(open(agent_cfg_path, "r"))

    def make_env():
        return gym.make(env_id, **env_cfg)

    envs = gym.vector.SyncVectorEnv([make_env])
    agent = make_agent(agent_cfg, envs).to(device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device)
    agent.load_state_dict(checkpoint["model"])
    agent.train()

    agent_phase1 = copy.deepcopy(agent)
    agent_phase1.eval()
    for p in agent_phase1.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo_cfg.get("learning_rate", 1e-4), eps=1e-5)
    buffer = OnlineBuffer(max_size=buffer_size, drop_easy_first=buffer_drop_easy_first)

    csv_path = os.path.join(output_dir, "phase2_multisample_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "batch", "case_start", "case_end", "trajectories", "dice_mean", "dice_std", "accept_rate",
        "avg_steps", "pg_loss", "v_loss", "entropy", "kl"
    ])

    case_stats = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_batches = (num_test_cases + samples_per_batch - 1) // samples_per_batch
    for batch_idx in range(num_batches):
        start_case = batch_idx * samples_per_batch
        end_case = min(start_case + samples_per_batch, num_test_cases)
        batch_dice = []
        batch_accepted = []
        batch_steps = []
        trajectories_batch = []
        terminal_rewards_batch = []

        for case_idx in range(start_case, end_case):
            for traj_idx in range(trajectories_per_sample):
                reset_seed = seed + case_idx * (trajectories_per_sample + 1) + traj_idx
                result = run_episode(
                    agent, envs, device, max_steps=max_steps,
                    store_trajectory=True, reset_seed=reset_seed,
                )
                traj = result.get("trajectory")
                if traj is None:
                    continue
                num_steps = result["num_steps"]
                final_mask = result["final_mask"]
                info = result.get("final_info", {})

                gt = _get_ground_truth_from_info(info)
                if gt is not None and final_mask is not None:
                    if isinstance(gt, np.ndarray):
                        gt_t = torch.tensor(gt, dtype=torch.float32, device=final_mask.device)
                        if gt_t.dim() == 2:
                            gt_t = gt_t.unsqueeze(0).unsqueeze(0)
                    else:
                        gt_t = gt
                    dice = compute_dice_score(final_mask, gt_t)
                else:
                    dice = 0.0

                accepted = simulate_physician_feedback(dice, threshold=acceptance_threshold)
                term_r = compute_terminal_reward(accepted, num_steps, prompt_penalty)

                batch_dice.append(dice)
                batch_accepted.append(accepted)
                batch_steps.append(num_steps)
                trajectories_batch.append(traj)
                terminal_rewards_batch.append(term_r)

                case_stats.append({
                    "case_id": case_idx,
                    "traj_idx": traj_idx,
                    "dice": dice,
                    "accepted": accepted,
                    "num_steps": num_steps,
                    "terminal_reward": term_r,
                })

        dice_mean = float(np.mean(batch_dice)) if batch_dice else 0.0
        dice_std = float(np.std(batch_dice)) if len(batch_dice) > 1 else 0.0
        accept_rate = float(np.mean(batch_accepted)) if batch_accepted else 0.0
        avg_steps = float(np.mean(batch_steps)) if batch_steps else 0.0

        buffer.add_batch(trajectories_batch, terminal_rewards_batch)

        n_sample = buffer_sample_size if buffer_sample_size is not None else len(buffer)
        sampled = buffer.sample(n_sample)
        update_metrics = None
        if len(sampled) >= 1:
            batch_data = flatten_trajectories_to_batch(sampled, device, gamma=gamma, gae_lambda=gae_lambda)
            for _ in range(update_epochs):
                update_metrics = ppo_update_online(
                    agent, agent_phase1, optimizer, batch_data, ppo_cfg, device
                )

        row = [
            batch_idx, start_case, end_case, len(trajectories_batch),
            f"{dice_mean:.4f}", f"{dice_std:.4f}", f"{accept_rate:.4f}", f"{avg_steps:.2f}",
            f"{update_metrics['pg_loss']:.6f}" if update_metrics else "",
            f"{update_metrics['v_loss']:.6f}" if update_metrics else "",
            f"{update_metrics['entropy']:.6f}" if update_metrics else "",
            f"{update_metrics['kl']:.6f}" if update_metrics else "",
        ]
        writer.writerow(row)
        csv_file.flush()

        if verbose:
            print(f"Phase2 multisample batch {batch_idx}: trajectories={len(trajectories_batch)} dice={dice_mean:.4f}±{dice_std:.4f} accept={accept_rate:.2%}")

    csv_file.close()

    if save_results:
        agent_path = os.path.join(output_dir, "phase2_multisample_agent.pth")
        torch.save({"model": agent.state_dict(), "config": cfg}, agent_path)
        case_csv = os.path.join(output_dir, "phase2_multisample_cases.csv")
        with open(case_csv, "w", newline="") as f:
            cw = csv.DictWriter(f, fieldnames=["case_id", "traj_idx", "dice", "accepted", "num_steps", "terminal_reward"])
            cw.writeheader()
            cw.writerows(case_stats)
        if verbose:
            print(f"Saved agent to {agent_path}, case stats to {case_csv}")

    envs.close()
    return output_dir, case_stats


def main():
    parser = argparse.ArgumentParser(description="Phase II Multi-Sample PPO")
    parser.add_argument("--config", type=str, default="phase2/configs/phase2_config_multisample.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    run_multisample_learning(args.config, args.output_dir)


if __name__ == "__main__":
    main()
