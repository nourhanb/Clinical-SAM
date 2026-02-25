"""
Episode runner for getting predictions from the agent
"""
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional


def load_obs_to_tensor(obs: Dict, device: torch.device) -> Dict:
    """Convert numpy observations to torch tensors."""
    obs_tensor = {}
    for key in obs.keys():
        if isinstance(obs[key], np.ndarray):
            obs_tensor[key] = torch.Tensor(obs[key]).to(device)
        else:
            obs_tensor[key] = obs[key]
    return obs_tensor


def run_episode(
    agent: torch.nn.Module,
    env: gym.Env,
    device: torch.device,
    max_steps: int = 50,
    store_trajectory: bool = False,
    reset_seed: int = None
) -> Dict:
    """
    Run a complete episode with the agent.
    
    Args:
        agent: The RL agent
        env: Gymnasium environment
        device: Torch device
        max_steps: Maximum steps per episode
        store_trajectory: Whether to store full trajectory (for PPO updates)
        reset_seed: Optional seed for deterministic environment reset
    
    Returns:
        Dictionary containing:
            - final_mask: Final predicted mask
            - total_reward: Cumulative reward
            - num_steps: Number of steps taken
            - trajectory: (Optional) Full trajectory data for PPO
    """
    if reset_seed is not None:
        obs, info = env.reset(seed=reset_seed)
    else:
        obs, info = env.reset()
    obs = load_obs_to_tensor(obs, device)
    
    # Storage for trajectory
    if store_trajectory:
        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
    
    total_reward = 0.0
    done = False
    step = 0
    actions_taken = []  # Debug: track actions
    last_info = info  # Initialize last_info in case loop doesn't run
    
    while not done and step < max_steps:
        # Get action from agent
        with torch.no_grad():
            action, log_prob, _, value = agent.get_action_and_value(obs)
            action_val = action.cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
            actions_taken.append(action_val)
        
        # Store trajectory data
        if store_trajectory:
            # Handle different types of observation values
            obs_cpu = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    obs_cpu[k] = v.cpu()
                else:
                    obs_cpu[k] = v  # Keep non-tensor values as is
            observations.append(obs_cpu)
            actions.append(action.cpu())
            log_probs.append(log_prob.cpu())
            values.append(value.cpu())
        
        # Take action in environment
        raw_next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated[0] or truncated[0]
        
        if store_trajectory:
            rewards.append(reward[0])
            dones.append(done)
        
        total_reward += reward[0]
        obs = load_obs_to_tensor(raw_next_obs, device)
        step += 1
        
        # Store the last info (for final mask extraction)
        last_info = info
    
    # Debug: Store actions for debugging
    if not hasattr(run_episode, '_debug_count'):
        run_episode._debug_count = 0
    run_episode._debug_count += 1
    if run_episode._debug_count <= 1:  # Only for first episode
        print(f"[DEBUG] Episode actions: {actions_taken[:10]}... (total: {len(actions_taken)} steps)")
    
    # Debug: Check episode status
    if step == 0:
        print(f"WARNING: Episode completed with 0 steps! max_steps={max_steps}, done={done}")
    
    # Debug: Check if any actions were taken (for first episode only)
    if hasattr(env, 'single_action_space'):
        action_space_size = env.single_action_space.n
    else:
        action_space_size = getattr(env, 'action_space', {}).n if hasattr(env, 'action_space') else 'unknown'
    
    # Check info for input points to see if agent took actions
    final_info_unwrapped = last_info
    if isinstance(last_info, dict):
        # Try to unwrap VectorEnv info
        for key in ['last_input_points', '_last_input_points']:
            if key in last_info:
                points = last_info[key]
                if isinstance(points, np.ndarray):
                    if points.dtype == object and points.size > 0:
                        points = points[0] if len(points) > 0 else points.item()
                    elif points.size > 0:
                        points = points[0] if len(points.shape) > 0 else points
                if isinstance(points, (list, np.ndarray)) and len(points) > 0:
                    break
    
    # Get final mask from info (VectorEnv wraps it)
    # Try final_info first (episode end info), then regular info
    final_mask = None
    if 'final_info' in last_info:
        final_info_array = last_info['final_info']
        if isinstance(final_info_array, np.ndarray) and final_info_array.size > 0:
            final_info_dict = final_info_array[0]
            if isinstance(final_info_dict, dict):
                # Check explicitly for None to avoid numpy array boolean error
                final_mask = final_info_dict.get('current_mask')
                if final_mask is None:
                    final_mask = final_info_dict.get('sam_pred_mask')
    
    # Fallback to regular info
    if final_mask is None:
        final_mask = last_info.get('current_mask', None)
    if final_mask is None:
        # Try alternative key
        final_mask = last_info.get('sam_pred_mask', None)
    
    if final_mask is not None:
        # VectorEnv wraps values in numpy arrays - unwrap if needed
        if isinstance(final_mask, np.ndarray):
            # Handle potential object dtype (VectorEnv wraps dicts/arrays)
            if final_mask.dtype == object:
                # Extract first element if it's wrapped
                if final_mask.size == 1:
                    final_mask = final_mask.item()
                else:
                    final_mask = final_mask[0]
            # If it's a regular array but has an extra dimension from VectorEnv
            elif len(final_mask.shape) > 2 and final_mask.shape[0] == 1:
                # VectorEnv added a batch dimension: (1, H, W) -> (H, W)
                final_mask = final_mask[0]
            
            # Convert to tensor
            if isinstance(final_mask, np.ndarray):
                final_mask = torch.from_numpy(final_mask.astype(np.float32)).to(device)
            else:
                final_mask = torch.tensor(final_mask, dtype=torch.float32).to(device)
        elif isinstance(final_mask, torch.Tensor):
            # If tensor has extra batch dimension from VectorEnv, remove it
            if len(final_mask.shape) > 2 and final_mask.shape[0] == 1:
                final_mask = final_mask[0]
            final_mask = final_mask.to(device)
    
    # Debug: Check mask extraction for first episode
    if not hasattr(run_episode, '_debug_mask_count'):
        run_episode._debug_mask_count = 0
    run_episode._debug_mask_count += 1
    if run_episode._debug_mask_count <= 1:
        print(f"[DEBUG] Mask extraction: final_mask={final_mask is not None}, shape={final_mask.shape if final_mask is not None else 'None'}, sum={final_mask.sum().item() if final_mask is not None else 'N/A'}")
        print(f"[DEBUG] Info keys: {list(last_info.keys())}")
        if 'current_mask' in last_info:
            cm = last_info['current_mask']
            print(f"[DEBUG] current_mask type: {type(cm)}, shape: {getattr(cm, 'shape', 'no shape')}, dtype: {getattr(cm, 'dtype', 'no dtype')}")
        if 'final_info' in last_info:
            print(f"[DEBUG] final_info present: {type(last_info['final_info'])}")
    
    result = {
        'final_mask': final_mask,
        'total_reward': total_reward,
        'num_steps': step,
        'final_info': last_info,
        'actions_taken': actions_taken  # Debug: include actions
    }
    
    if store_trajectory:
        result['trajectory'] = {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'dones': dones
        }
    
    return result


def get_mask_from_episode(
    agent: torch.nn.Module,
    env: gym.Env,
    image: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, Dict]:
    """
    Simplified function to get just the final mask from an episode.
    
    Args:
        agent: The RL agent
        env: Gymnasium environment
        image: Input image
        device: Torch device
    
    Returns:
        final_mask: The predicted segmentation mask
        info: Additional information dictionary
    """
    result = run_episode(agent, env, device, store_trajectory=False)
    return result['final_mask'], result['final_info']

