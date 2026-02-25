"""
Phase II Online PPO: Prequential test-then-learn with physician feedback.

Paper §2.3: SAM frozen; only prompting policy π_θ updated online. For each test case:
policy proposes prompts → SAM updates mask → physician gives y ∈ {0,1}. Terminal reward
r = (2*y - 1) - η*K. Update every B cases with PPO from a buffer; KL penalty toward Phase I.
"""
import os
import sys
import csv
import copy
import argparse
from collections import deque

# Set CuBLAS deterministic before torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml

# Project root
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


def _unwrap_info(info, key, default=None):
    """Unwrap a value from VectorEnv info (may be array of shape (1,) or nested)."""
    if key not in info:
        return default
    v = info[key]
    if isinstance(v, np.ndarray):
        if v.dtype == object and v.size >= 1:
            return v.flat[0]
        if v.size >= 1:
            return v.flat[0]
        return default
    return v


def _get_ground_truth_from_info(info):
    """Extract ground truth mask from env info (VectorEnv-aware)."""
    # Prefer final_info for end-of-episode
    if "final_info" in info:
        fi = info["final_info"]
        if isinstance(fi, np.ndarray) and fi.size > 0:
            d = fi.flat[0]
            if isinstance(d, dict) and "ground_truth_mask" in d:
                return d["ground_truth_mask"]
    if "ground_truth_mask" in info:
        return _unwrap_info(info, "ground_truth_mask")
    return None


def compute_terminal_reward(accepted: int, num_prompts: int, prompt_penalty: float) -> float:
    """r = (2*accept - 1) - η*K."""
    return (2.0 * accepted - 1.0) - prompt_penalty * num_prompts


def trajectory_with_terminal_reward(trajectory: dict, terminal_reward: float) -> dict:
    """Set rewards to 0 except last step; last step = terminal_reward."""
    L = len(trajectory["rewards"])
    traj = {k: list(v) for k, v in trajectory.items()}
    traj["rewards"] = [0.0] * (L - 1) + [terminal_reward]
    traj["dones"] = [False] * (L - 1) + [True]
    return traj


# Keys the agent actually uses (implicit_agent.parse_obs)
_OBS_KEYS_FOR_AGENT = ("sam_image_embeddings", "sam_pred_mask_prob")


def stack_observations(obs_list):
    """Stack observation dicts into batched tensors. Only stacks agent keys; converts numpy to tensor."""
    if not obs_list:
        return None
    stacked = {}
    for k in _OBS_KEYS_FOR_AGENT:
        if k not in obs_list[0]:
            continue
        out = []
        for o in obs_list:
            v = o[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if not isinstance(v, torch.Tensor):
                break
            out.append(v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v)
        if len(out) == len(obs_list):
            stacked[k] = torch.stack(out, dim=0)
    if len(stacked) < len(_OBS_KEYS_FOR_AGENT):
        missing = set(_OBS_KEYS_FOR_AGENT) - set(stacked.keys())
        raise ValueError(
            f"stack_observations: missing or non-numeric keys for agent: {missing}. "
            "Each obs step must have tensor or ndarray for sam_image_embeddings and sam_pred_mask_prob."
        )
    return stacked


def flatten_trajectories_to_batch(trajectories, device, gamma=0.99, gae_lambda=0.95):
    """
    trajectories: list of dicts with observations, actions, log_probs, values, rewards, dones.
    Caller must have set rewards/dones to terminal-reward form. Compute GAE per traj then concat.
    """
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_dones = []
    for traj in trajectories:
        all_obs.append(traj["observations"])
        all_actions.append(torch.stack(traj["actions"]))
        all_log_probs.append(torch.stack(traj["log_probs"]))
        all_values.append(torch.stack(traj["values"]))
        all_rewards.append(np.array(traj["rewards"], dtype=np.float64))
        all_dones.append(np.array(traj["dones"], dtype=bool))

    all_advantages = []
    all_returns = []
    for r, v, d in zip(all_rewards, all_values, all_dones):
        v_np = v.detach().cpu().numpy().flatten()
        adv, ret = compute_gae(list(r), list(v_np), list(d), gamma=gamma, gae_lambda=gae_lambda)
        all_advantages.append(adv)
        all_returns.append(ret)

    b_actions = torch.cat(all_actions, dim=0).to(device)
    b_log_probs = torch.cat(all_log_probs, dim=0).to(device)
    b_values = torch.cat(all_values, dim=0).to(device)
    b_advantages = np.concatenate(all_advantages, axis=0)
    b_returns = torch.tensor(np.concatenate(all_returns, axis=0), dtype=torch.float32, device=device)

    flat_obs_list = []
    for obs_list in all_obs:
        for o in obs_list:
            flat_obs_list.append(o)
    b_obs = stack_observations(flat_obs_list)
    if b_obs is not None:
        for k in b_obs:
            b_obs[k] = b_obs[k].to(device)

    return {
        "b_obs": b_obs,
        "b_actions": b_actions,
        "b_log_probs": b_log_probs,
        "b_values": b_values,
        "b_advantages": b_advantages,
        "b_returns": b_returns,
    }


class OnlineBuffer:
    """FIFO buffer of trajectories (with terminal rewards already applied)."""

    def __init__(self, max_size: int, drop_easy_first: bool = False):
        self.max_size = max_size
        self.drop_easy_first = drop_easy_first
        self._buffer = deque(maxlen=max_size)
        self._terminal_rewards = deque(maxlen=max_size)

    def add(self, trajectory: dict, terminal_reward: float):
        traj = trajectory_with_terminal_reward(trajectory, terminal_reward)
        if len(self._buffer) >= self.max_size and self.drop_easy_first and terminal_reward > 0:
            # Optionally drop an easy (positive reward) sample; for now we just FIFO
            pass
        self._buffer.append(traj)
        self._terminal_rewards.append(terminal_reward)

    def add_batch(self, trajectories: list, terminal_rewards: list):
        for t, r in zip(trajectories, terminal_rewards):
            self.add(t, r)

    def sample(self, n: int = None):
        """Return up to n trajectories (or all if n is None)."""
        if n is None or n >= len(self._buffer):
            return list(self._buffer)
        inds = np.random.choice(len(self._buffer), size=min(n, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in inds]

    def __len__(self):
        return len(self._buffer)


def ppo_update_online(
    agent,
    agent_phase1,
    optimizer,
    batch_data,
    ppo_cfg,
    device,
):
    """One PPO update with KL penalty toward agent_phase1."""
    b_obs = batch_data["b_obs"]
    b_actions = batch_data["b_actions"].long()
    b_log_probs = batch_data["b_log_probs"]
    b_values = batch_data["b_values"]
    b_advantages = batch_data["b_advantages"]
    b_returns = batch_data["b_returns"]

    lr = float(ppo_cfg.get("learning_rate", 1e-4))
    clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
    value_loss_coef = ppo_cfg.get("value_loss_coef", 0.5)
    entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
    kl_coef = ppo_cfg.get("kl_coef", 0.1)
    max_grad_norm = ppo_cfg.get("max_grad_norm", 10.0)
    norm_adv = True

    for g in optimizer.param_groups:
        g["lr"] = lr

    N = b_actions.shape[0]
    b_inds = np.arange(N)
    np.random.shuffle(b_inds)
    minibatch_size = min(64, N)
    if minibatch_size < 1:
        return None

    def get_obs_at_inds(obs_dict, inds):
        out = {}
        for k, v in obs_dict.items():
            out[k] = v[inds]
        return out

    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_minibatches = 0

    for start in range(0, N, minibatch_size):
        end = min(start + minibatch_size, N)
        mb_inds = b_inds[start:end]
        mb_obs = get_obs_at_inds(b_obs, mb_inds)
        mb_advantages = torch.tensor(b_advantages[mb_inds], dtype=torch.float32, device=device)
        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        mb_actions = b_actions[mb_inds]
        # Ensure 1D action for agent (trajectory may store (1,) per step -> (N,1) after cat)
        if mb_actions.dim() > 1:
            mb_actions = mb_actions.squeeze(-1)
        _, new_log_prob, entropy, new_value = agent.get_action_and_value(mb_obs, mb_actions)
        new_value = new_value.view(-1)
        new_log_prob = new_log_prob.view(-1)
        entropy = entropy.view(-1)

        mb_len = mb_advantages.shape[0]
        if new_log_prob.shape[0] != mb_len:
            new_log_prob = new_log_prob.view(mb_len, -1).sum(dim=1) if new_log_prob.numel() >= mb_len else new_log_prob[:mb_len]
        if new_value.shape[0] != mb_len:
            new_value = new_value[:mb_len]
        if entropy.shape[0] != mb_len:
            entropy = entropy[:mb_len]

        # KL toward Phase I
        with torch.no_grad():
            old_logits = agent_phase1.get_logits(mb_obs)
        new_logits = agent.get_logits(mb_obs)
        kl_loss = compute_kl_divergence(new_logits, old_logits, reduction="batchmean")

        old_log_probs_mb = b_log_probs[mb_inds].view(-1)
        if old_log_probs_mb.shape[0] != mb_len:
            old_log_probs_mb = old_log_probs_mb[:mb_len]
        ratio = (new_log_prob - old_log_probs_mb).exp()
        if ratio.shape[0] != mb_len:
            ratio = ratio[:mb_len]
        pg1 = -mb_advantages * ratio
        pg2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        pg_loss = torch.max(pg1, pg2).mean()

        b_returns_mb = b_returns[mb_inds].view(-1)
        if b_returns_mb.shape[0] != mb_len:
            b_returns_mb = b_returns_mb[:mb_len]
        v_loss = 0.5 * ((new_value - b_returns_mb) ** 2).mean()
        loss = pg_loss + value_loss_coef * v_loss - entropy_coef * entropy.mean() + kl_coef * kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        total_pg_loss += pg_loss.item()
        total_v_loss += v_loss.item()
        total_entropy += entropy.mean().item()
        total_kl += kl_loss.item()
        n_minibatches += 1

    if n_minibatches == 0:
        return None
    return {
        "pg_loss": total_pg_loss / n_minibatches,
        "v_loss": total_v_loss / n_minibatches,
        "entropy": total_entropy / n_minibatches,
        "kl": total_kl / n_minibatches,
    }


def run_online_learning(config_path: str, output_dir: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if output_dir is None:
        output_dir = cfg.get("output_dir", "phase2/results_online")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    env_id = cfg.get("env_id", "SamSegEnv-v0")
    env_cfg_path = cfg.get("env_cfg_path")
    agent_cfg_path = cfg.get("agent_cfg_path")
    phase1_checkpoint = cfg.get("phase1_checkpoint")
    num_test_cases = cfg.get("num_test_cases", 26)
    seed = cfg.get("seed", 42)
    samples_per_batch = cfg.get("samples_per_batch", 5)
    update_epochs = cfg.get("update_epochs", 1)
    max_steps = cfg.get("max_steps", 50)
    acceptance_threshold = cfg.get("acceptance_threshold", 0.75)
    prompt_penalty = cfg.get("prompt_penalty", 0.02)
    buffer_size = cfg.get("buffer_size", 32)
    buffer_drop_easy_first = cfg.get("buffer_drop_easy_first", False)
    buffer_sample_size = cfg.get("buffer_sample_size")
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

    lr = float(ppo_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    buffer = OnlineBuffer(max_size=buffer_size, drop_easy_first=buffer_drop_easy_first)

    # Prequential log: per-batch metrics before update
    csv_path = os.path.join(output_dir, "phase2_online_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "batch", "case_start", "case_end", "dice_mean", "dice_std", "accept_rate",
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

        for i in range(start_case, end_case):
            case_idx = i
            reset_seed = seed + case_idx
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
                "dice": dice,
                "accepted": accepted,
                "num_steps": num_steps,
                "terminal_reward": term_r,
            })

        # Prequential: log batch metrics before update
        dice_mean = float(np.mean(batch_dice)) if batch_dice else 0.0
        dice_std = float(np.std(batch_dice)) if len(batch_dice) > 1 else 0.0
        accept_rate = float(np.mean(batch_accepted)) if batch_accepted else 0.0
        avg_steps = float(np.mean(batch_steps)) if batch_steps else 0.0

        # Add to buffer
        buffer.add_batch(trajectories_batch, terminal_rewards_batch)

        # PPO update from buffer
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
            batch_idx, start_case, end_case, f"{dice_mean:.4f}", f"{dice_std:.4f}", f"{accept_rate:.4f}",
            f"{avg_steps:.2f}",
            f"{update_metrics['pg_loss']:.6f}" if update_metrics else "",
            f"{update_metrics['v_loss']:.6f}" if update_metrics else "",
            f"{update_metrics['entropy']:.6f}" if update_metrics else "",
            f"{update_metrics['kl']:.6f}" if update_metrics else "",
        ]
        writer.writerow(row)
        csv_file.flush()

        if verbose:
            print(f"Phase2 online batch {batch_idx}: dice={dice_mean:.4f}±{dice_std:.4f} accept={accept_rate:.2%} steps={avg_steps:.1f}")

    csv_file.close()

    if save_results:
        agent_path = os.path.join(output_dir, "phase2_online_agent.pth")
        torch.save({"model": agent.state_dict(), "config": cfg}, agent_path)
        case_csv = os.path.join(output_dir, "phase2_online_cases.csv")
        with open(case_csv, "w", newline="") as f:
            cw = csv.DictWriter(f, fieldnames=["case_id", "dice", "accepted", "num_steps", "terminal_reward"])
            cw.writeheader()
            cw.writerows(case_stats)
        if verbose:
            print(f"Saved agent to {agent_path}, case stats to {case_csv}")

    envs.close()
    return output_dir, case_stats


def main():
    parser = argparse.ArgumentParser(description="Phase II Online PPO (prequential test-then-learn)")
    parser.add_argument("--config", type=str, default="phase2/configs/phase2_config_online.yaml", help="Config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    run_online_learning(args.config, args.output_dir)


if __name__ == "__main__":
    main()
