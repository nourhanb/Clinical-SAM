"""
Evaluation script for AlignSAM trained models.
Loads a trained checkpoint and evaluates the agent on the environment.
"""
import os
import argparse
from dataclasses import dataclass

# Set environment variable for CuBLAS deterministic behavior BEFORE importing torch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models import make_agent
import custom_gym_implns


def load_obs_to_tensor(obs, device):
    obs_tensor = dict()
    for key in obs.keys():
        if isinstance(obs[key], np.ndarray):
            obs_tensor[key] = torch.Tensor(obs[key]).to(device)
        else:
            obs_tensor[key] = obs[key]
    return obs_tensor


def evaluate(args):
    # Load configs
    env_cfg = yaml.safe_load(open(args.env_cfg_path, "r"))
    agent_cfg = yaml.safe_load(open(args.agent_cfg_path, "r"))
    
    # Device setup
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory for videos if needed
    if args.save_videos:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Environment setup - use same pattern as train_sam_align_ppo.py
    def make_env():
        env = gym.make(args.env_id, **env_cfg)
        if args.save_videos:
            env = gym.wrappers.RecordVideo(
                env, 
                args.output_dir, 
                episode_trigger=lambda x: True  # Record all episodes
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    # Wrap in a vector env for compatibility with agent
    envs = gym.vector.SyncVectorEnv([make_env])
    
    # Create agent and load checkpoint
    agent = make_agent(agent_cfg, envs).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    
    print(f"Loaded checkpoint from: {args.checkpoint_path}")
    
    # Evaluation loop
    all_rewards = []
    all_episode_lengths = []
    all_final_ious = []
    
    for episode in tqdm(range(args.num_episodes), desc="Evaluating"):
        # Pass seed to ensure deterministic environment reset
        obs, info = envs.reset(seed=args.seed + episode)
        obs = load_obs_to_tensor(obs, device)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            
            raw_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = terminated[0] or truncated[0]
            
            episode_reward += reward[0]
            obs = load_obs_to_tensor(raw_next_obs, device)
            step += 1
            
            # Store the last info (final step info contains dice_score)
            last_info = info
        
        # Get episode statistics
        if "episode" in last_info:
            ep_info = last_info["episode"]
            all_rewards.append(ep_info["r"][0])
            all_episode_lengths.append(ep_info["l"][0])
        else:
            all_rewards.append(episode_reward)
            all_episode_lengths.append(step)
        
        # Debug: Print info keys for first episode
        if episode == 0 and args.num_episodes > 0:
            print(f"\n[DEBUG] Episode {episode} info keys: {last_info.keys()}")
            for key, val in last_info.items():
                if isinstance(val, (np.ndarray, list, tuple)):
                    print(f"  {key}: type={type(val)}, shape/len={getattr(val, 'shape', len(val))}")
                else:
                    print(f"  {key}: {val}")
        
        # Get final IoU / Dice / dice_score from info if available
        # The vector env wraps values in numpy arrays, so we need to unwrap carefully
        metric_val = None
        
        # Try to get from final_info first (the proper location for episode end info)
        if "final_info" in last_info:
            final_info_array = last_info["final_info"]
            if isinstance(final_info_array, np.ndarray) and final_info_array.size > 0:
                final_info_dict = final_info_array[0]  # Get first env's info
                if isinstance(final_info_dict, dict):
                    if "dice_score" in final_info_dict:
                        metric_val = final_info_dict["dice_score"]
                    elif "dice" in final_info_dict:
                        metric_val = final_info_dict["dice"]
                    elif "iou" in final_info_dict:
                        metric_val = final_info_dict["iou"]
        
        # Fallback: try direct keys (though these are also wrapped in arrays)
        if metric_val is None:
            for metric_key in ["dice_score", "dice", "iou"]:
                if metric_key in last_info:
                    val = last_info[metric_key]
                    if isinstance(val, np.ndarray) and val.size > 0:
                        metric_val = val[0]
                        break
                    elif isinstance(val, (int, float)):
                        metric_val = val
                        break
        
        # Convert to float and append
        if metric_val is not None:
            try:
                metric_val = float(metric_val)
                all_final_ious.append(metric_val)
                if episode == 0:
                    print(f"[DEBUG] Extracted dice_score: {metric_val}")
            except Exception as e:
                if episode == 0:
                    print(f"[DEBUG] Failed to convert metric to float: {e}, type: {type(metric_val)}, value: {metric_val}")
        else:
            if episode == 0:
                print(f"[DEBUG] No metric found in info")
    envs.close()
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"Mean episode length: {np.mean(all_episode_lengths):.2f} ± {np.std(all_episode_lengths):.2f}")
    if all_final_ious:
        print(f"Mean Dice Score: {np.mean(all_final_ious):.4f} ± {np.std(all_final_ious):.4f}")
        print(f"Min Dice Score: {np.min(all_final_ious):.4f}")
        print(f"Max Dice Score: {np.max(all_final_ious):.4f}")
    else:
        print("WARNING: No Dice/IoU scores were collected!")
    print("=" * 50)
    
    # Save results to file
    if args.save_results:
        results_path = os.path.join(args.output_dir, "eval_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Number of episodes: {args.num_episodes}\n")
            f.write(f"Mean reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}\n")
            f.write(f"Mean episode length: {np.mean(all_episode_lengths):.2f} ± {np.std(all_episode_lengths):.2f}\n")
            if all_final_ious:
                f.write(f"Mean Dice Score: {np.mean(all_final_ious):.4f} ± {np.std(all_final_ious):.4f}\n")
                f.write(f"Min Dice Score: {np.min(all_final_ious):.4f}\n")
                f.write(f"Max Dice Score: {np.max(all_final_ious):.4f}\n")
        print(f"Results saved to: {results_path}")
    
    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_episode_length": np.mean(all_episode_lengths),
        "mean_iou": np.mean(all_final_ious) if all_final_ious else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AlignSAM trained model")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the checkpoint file (.pth)"
    )
    parser.add_argument(
        "--env_cfg_path", 
        type=str, 
        default="configs/envs/repvit_sam_medical_test.yaml",
        help="Path to environment config"
    )
    parser.add_argument(
        "--agent_cfg_path", 
        type=str, 
        default="configs/agents/implicit_agent.yaml",
        help="Path to agent config"
    )
    parser.add_argument(
        "--env_id", 
        type=str, 
        default="SamSegEnv-v0",
        help="Environment ID"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--save_videos", 
        action="store_true",
        help="Whether to save video recordings of episodes"
    )
    parser.add_argument(
        "--save_results", 
        action="store_true",
        help="Whether to save evaluation results to a file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="eval_output",
        help="Directory to save videos and results"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Enable deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where possible
        # Note: Some operations (like cumsum) don't have deterministic implementations
        # but we use warn_only=True to continue execution
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    evaluate(args)
