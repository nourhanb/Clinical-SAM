"""
Utility functions for test-time adaptation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice score between prediction and target.
    
    Args:
        pred: Predicted mask (H, W) or (B, C, H, W) with probabilities
        target: Ground truth mask (H, W) or (B, C, H, W) with binary values
        threshold: Threshold for binarizing prediction
    
    Returns:
        Dice score (float)
    """
    # Handle None masks
    if pred is None or target is None:
        return 0.0
    
    # Ensure both are tensors
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    
    # Handle different shapes: (H, W) -> (1, 1, H, W) for consistency
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif target.dim() == 3:
        target = target.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    # Ensure same shape
    if pred.shape != target.shape:
        # Try to broadcast or resize
        if pred.shape[2:] != target.shape[2:]:
            # Resize target to match pred
            target = torch.nn.functional.interpolate(
                target, size=pred.shape[2:], mode='nearest'
            )
        # Handle channel mismatch
        if pred.shape[1] != target.shape[1]:
            if pred.shape[1] == 1 and target.shape[1] > 1:
                # Take first channel of target
                target = target[:, 0:1, :, :]
            elif target.shape[1] == 1 and pred.shape[1] > 1:
                # Take first channel of pred
                pred = pred[:, 0:1, :, :]
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


def simulate_physician_feedback(dice_score: float, threshold: float = 0.70) -> int:
    """
    Simulate binary physician feedback based on Dice score.
    
    Args:
        dice_score: Dice coefficient between prediction and ground truth
        threshold: Acceptance threshold (default: 0.70)
    
    Returns:
        1 if accepted (Dice >= threshold), 0 if rejected
    """
    return 1 if dice_score >= threshold else 0


def compute_kl_divergence(
    new_logits: torch.Tensor, 
    old_logits: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    Compute KL divergence between new and old policy distributions.
    Used for preventing catastrophic forgetting.
    
    Args:
        new_logits: Logits from current policy
        old_logits: Logits from Phase 1 policy
        reduction: Reduction method
    
    Returns:
        KL divergence loss
    """
    log_new_probs = F.log_softmax(new_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)
    
    kl_loss = F.kl_div(log_new_probs, old_probs, reduction=reduction)
    return kl_loss


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards per step
        values: List of value estimates per step
        dones: List of done flags per step
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Array of advantages
        returns: Array of returns
    """
    advantages = []
    returns = []
    gae = 0
    
    # Reverse iteration
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    advantages = np.array(advantages)
    returns = np.array(returns)
    
    return advantages, returns


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss.
    
    Args:
        pred: Predicted mask with probabilities
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy loss.
    
    Args:
        pred: Predicted mask with probabilities
        target: Ground truth mask
    
    Returns:
        BCE loss
    """
    return F.binary_cross_entropy(pred, target, reduction='mean')


class AdaptationTracker:
    """Track adaptation statistics during test-time adaptation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.case_stats = []
        self.global_stats = {
            'total_cases': 0,
            'initial_accepted': 0,
            'final_accepted': 0,
            'total_adaptations': 0,
            'avg_dice_initial': 0.0,
            'avg_dice_final': 0.0,
            'avg_dice_improvement': 0.0
        }
    
    def add_case(self, case_dict: Dict):
        """
        Add results for a single case.
        
        Args:
            case_dict: Dictionary with keys:
                - case_id: Identifier
                - dice_initial: Initial Dice score
                - dice_final: Final Dice score
                - feedback_initial: Initial feedback (0/1)
                - feedback_final: Final feedback (0/1)
                - num_adaptations: Number of adaptation steps
        """
        self.case_stats.append(case_dict)
        self._update_global_stats()
    
    def _update_global_stats(self):
        """Update global statistics."""
        if not self.case_stats:
            return
        
        self.global_stats['total_cases'] = len(self.case_stats)
        self.global_stats['initial_accepted'] = sum(c['feedback_initial'] for c in self.case_stats)
        self.global_stats['final_accepted'] = sum(c['feedback_final'] for c in self.case_stats)
        self.global_stats['total_adaptations'] = sum(c['num_adaptations'] for c in self.case_stats)
        
        self.global_stats['avg_dice_initial'] = np.mean([c['dice_initial'] for c in self.case_stats])
        self.global_stats['avg_dice_final'] = np.mean([c['dice_final'] for c in self.case_stats])
        self.global_stats['avg_dice_improvement'] = (
            self.global_stats['avg_dice_final'] - self.global_stats['avg_dice_initial']
        )
    
    def get_summary(self) -> str:
        """Get formatted summary string (matches evaluate.py format)."""
        stats = self.global_stats
        
        # Check if we have any valid cases
        if not self.case_stats:
            return "\n" + "="*80 + "\nERROR: No valid cases processed! All cases were skipped.\n" + "="*80 + "\n"
        
        # Compute detailed statistics
        dice_initial_list = [c['dice_initial'] for c in self.case_stats]
        dice_final_list = [c['dice_final'] for c in self.case_stats]
        dice_improvement_list = [c['dice_final'] - c['dice_initial'] for c in self.case_stats]
        
        dice_initial_mean = np.mean(dice_initial_list)
        dice_initial_std = np.std(dice_initial_list)
        dice_final_mean = np.mean(dice_final_list)
        dice_final_std = np.std(dice_final_list)
        dice_improvement_mean = np.mean(dice_improvement_list)
        dice_improvement_std = np.std(dice_improvement_list)
        
        summary = f"""
{'='*80}
PHASE 2 TEST-TIME ADAPTATION RESULTS
{'='*80}
Number of cases: {stats['total_cases']}

DICE SCORES (Initial - Before Adaptation):
  Mean Dice Score: {dice_initial_mean:.4f} ± {dice_initial_std:.4f}
  Min Dice Score: {min(dice_initial_list):.4f}
  Max Dice Score: {max(dice_initial_list):.4f}

DICE SCORES (Final - After Adaptation):
  Mean Dice Score: {dice_final_mean:.4f} ± {dice_final_std:.4f}
  Min Dice Score: {min(dice_final_list):.4f}
  Max Dice Score: {max(dice_final_list):.4f}

IMPROVEMENT:
  Mean Dice Improvement: {dice_improvement_mean:.4f} ± {dice_improvement_std:.4f}
  Relative Improvement: {100*dice_improvement_mean/max(dice_initial_mean,0.01):.1f}%

PHYSICIAN FEEDBACK (Acceptance Rate):
  Initial (Before Adaptation): {stats['initial_accepted']}/{stats['total_cases']} ({100*stats['initial_accepted']/max(stats['total_cases'],1):.1f}%)
  Final (After Adaptation): {stats['final_accepted']}/{stats['total_cases']} ({100*stats['final_accepted']/max(stats['total_cases'],1):.1f}%)
  Improvement: +{stats['final_accepted'] - stats['initial_accepted']} cases accepted

ADAPTATION STATISTICS:
  Total adaptations performed: {stats['total_adaptations']}
  Mean adaptations per case: {stats['total_adaptations']/max(stats['total_cases'],1):.2f}
  Cases that improved: {sum(1 for d in dice_improvement_list if d > 0)} ({100*sum(1 for d in dice_improvement_list if d > 0)/max(stats['total_cases'],1):.1f}%)
  Cases that degraded: {sum(1 for d in dice_improvement_list if d < 0)} ({100*sum(1 for d in dice_improvement_list if d < 0)/max(stats['total_cases'],1):.1f}%)
{'='*80}

COMPARISON WITH PHASE 1 (evaluate.py):
  Phase 1 Mean Dice: {dice_initial_mean:.4f} ± {dice_initial_std:.4f}
  Phase 2 Mean Dice: {dice_final_mean:.4f} ± {dice_final_std:.4f}
  Absolute Gain: +{dice_improvement_mean:.4f}
  Relative Gain: +{100*dice_improvement_mean/max(dice_initial_mean,0.01):.1f}%
{'='*80}
"""
        return summary
    
    def save_results(self, filepath: str):
        """Save results to CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.case_stats)
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
    
    def get_comparison_table(self, phase1_dice_mean: float = None, phase1_dice_std: float = None) -> str:
        """
        Get a formatted comparison table for paper/presentation.
        
        Args:
            phase1_dice_mean: Mean Dice from Phase 1 (evaluate.py)
            phase1_dice_std: Std Dice from Phase 1 (evaluate.py)
        
        Returns:
            Formatted markdown table
        """
        dice_initial_list = [c['dice_initial'] for c in self.case_stats]
        dice_final_list = [c['dice_final'] for c in self.case_stats]
        
        dice_initial_mean = np.mean(dice_initial_list)
        dice_initial_std = np.std(dice_initial_list)
        dice_final_mean = np.mean(dice_final_list)
        dice_final_std = np.std(dice_final_list)
        
        # Use provided Phase 1 stats or use initial as proxy
        if phase1_dice_mean is None:
            phase1_dice_mean = dice_initial_mean
            phase1_dice_std = dice_initial_std
        
        acceptance_initial = self.global_stats['initial_accepted'] / max(self.global_stats['total_cases'], 1)
        acceptance_final = self.global_stats['final_accepted'] / max(self.global_stats['total_cases'], 1)
        
        table = f"""
COMPARISON TABLE (for paper):
================================================================================
| Method                  | Mean Dice ± Std    | Min    | Max    | Acceptance Rate |
|-------------------------|--------------------|--------|--------|-----------------|
| Phase 1 (No Adaptation) | {phase1_dice_mean:.4f} ± {phase1_dice_std:.4f} | {min(dice_initial_list):.4f} | {max(dice_initial_list):.4f} | {100*acceptance_initial:.1f}%         |
| Phase 2 (With Adaptation)| {dice_final_mean:.4f} ± {dice_final_std:.4f} | {min(dice_final_list):.4f} | {max(dice_final_list):.4f} | {100*acceptance_final:.1f}%         |
| Improvement             | +{dice_final_mean-phase1_dice_mean:.4f}            | --     | --     | +{100*(acceptance_final-acceptance_initial):.1f}%          |
================================================================================
"""
        return table

