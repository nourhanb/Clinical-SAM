"""
Compare results between PPO and Supervised adaptation methods
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_results(ppo_path, supervised_path):
    """Load results from both methods."""
    ppo_df = pd.read_csv(ppo_path)
    sup_df = pd.read_csv(supervised_path)
    return ppo_df, sup_df


def compute_statistics(df, method_name):
    """Compute statistics for a results dataframe."""
    stats = {
        'method': method_name,
        'total_cases': len(df),
        'initial_acceptance_rate': df['feedback_initial'].mean(),
        'final_acceptance_rate': df['feedback_final'].mean(),
        'acceptance_improvement': df['feedback_final'].mean() - df['feedback_initial'].mean(),
        'mean_dice_initial': df['dice_initial'].mean(),
        'mean_dice_final': df['dice_final'].mean(),
        'mean_dice_improvement': (df['dice_final'] - df['dice_initial']).mean(),
        'std_dice_improvement': (df['dice_final'] - df['dice_initial']).std(),
        'total_adaptations': df['num_adaptations'].sum(),
        'mean_adaptations_per_case': df['num_adaptations'].mean(),
        'cases_improved': ((df['dice_final'] - df['dice_initial']) > 0).sum(),
        'cases_degraded': ((df['dice_final'] - df['dice_initial']) < 0).sum(),
    }
    return stats


def print_comparison(ppo_stats, sup_stats):
    """Print comparison table."""
    print("\n" + "="*80)
    print("PHASE 2 METHODS COMPARISON")
    print("="*80)
    print(f"{'Metric':<40} {'PPO':>15} {'Supervised':>15} {'Winner':>10}")
    print("-"*80)
    
    metrics = [
        ('Total Cases', 'total_cases', 'higher'),
        ('Initial Acceptance Rate', 'initial_acceptance_rate', 'higher'),
        ('Final Acceptance Rate', 'final_acceptance_rate', 'higher'),
        ('Acceptance Improvement', 'acceptance_improvement', 'higher'),
        ('Mean Dice Initial', 'mean_dice_initial', 'higher'),
        ('Mean Dice Final', 'mean_dice_final', 'higher'),
        ('Mean Dice Improvement', 'mean_dice_improvement', 'higher'),
        ('Std Dice Improvement', 'std_dice_improvement', 'lower'),
        ('Total Adaptations', 'total_adaptations', 'lower'),
        ('Mean Adaptations/Case', 'mean_adaptations_per_case', 'lower'),
        ('Cases Improved', 'cases_improved', 'higher'),
        ('Cases Degraded', 'cases_degraded', 'lower'),
    ]
    
    for metric_name, metric_key, better in metrics:
        ppo_val = ppo_stats[metric_key]
        sup_val = sup_stats[metric_key]
        
        # Format based on type
        if isinstance(ppo_val, float) and abs(ppo_val) < 10:
            ppo_str = f"{ppo_val:.4f}"
            sup_str = f"{sup_val:.4f}"
        else:
            ppo_str = f"{ppo_val:.2f}" if isinstance(ppo_val, float) else f"{ppo_val}"
            sup_str = f"{sup_val:.2f}" if isinstance(sup_val, float) else f"{sup_val}"
        
        # Determine winner
        if better == 'higher':
            winner = 'PPO' if ppo_val > sup_val else 'Supervised' if sup_val > ppo_val else 'Tie'
        else:
            winner = 'PPO' if ppo_val < sup_val else 'Supervised' if sup_val < ppo_val else 'Tie'
        
        print(f"{metric_name:<40} {ppo_str:>15} {sup_str:>15} {winner:>10}")
    
    print("="*80)


def plot_comparison(ppo_df, sup_df, output_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Dice scores distribution
    ax = axes[0, 0]
    ax.hist(ppo_df['dice_initial'], bins=20, alpha=0.5, label='Phase 1 (Initial)', color='gray')
    ax.hist(ppo_df['dice_final'], bins=20, alpha=0.5, label='PPO', color='blue')
    ax.hist(sup_df['dice_final'], bins=20, alpha=0.5, label='Supervised', color='green')
    ax.set_xlabel('Dice Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Dice Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dice improvement
    ax = axes[0, 1]
    ppo_improvement = ppo_df['dice_final'] - ppo_df['dice_initial']
    sup_improvement = sup_df['dice_final'] - sup_df['dice_initial']
    ax.hist(ppo_improvement, bins=20, alpha=0.5, label='PPO', color='blue')
    ax.hist(sup_improvement, bins=20, alpha=0.5, label='Supervised', color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.set_xlabel('Dice Improvement')
    ax.set_ylabel('Frequency')
    ax.set_title('Dice Score Improvement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Acceptance rates
    ax = axes[1, 0]
    methods = ['Phase 1\n(Initial)', 'PPO', 'Supervised']
    initial_rate = ppo_df['feedback_initial'].mean()
    ppo_rate = ppo_df['feedback_final'].mean()
    sup_rate = sup_df['feedback_final'].mean()
    rates = [initial_rate, ppo_rate, sup_rate]
    colors = ['gray', 'blue', 'green']
    bars = ax.bar(methods, rates, color=colors, alpha=0.7)
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Physician Acceptance Rate')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2%}', ha='center', va='bottom')
    
    # Plot 4: Number of adaptations
    ax = axes[1, 1]
    adaptation_data = [
        ppo_df['num_adaptations'].value_counts().sort_index(),
        sup_df['num_adaptations'].value_counts().sort_index()
    ]
    x = np.arange(max(ppo_df['num_adaptations'].max(), sup_df['num_adaptations'].max()) + 1)
    width = 0.35
    ppo_counts = [ppo_df[ppo_df['num_adaptations'] == i].shape[0] for i in x]
    sup_counts = [sup_df[sup_df['num_adaptations'] == i].shape[0] for i in x]
    
    ax.bar(x - width/2, ppo_counts, width, label='PPO', color='blue', alpha=0.7)
    ax.bar(x + width/2, sup_counts, width, label='Supervised', color='green', alpha=0.7)
    ax.set_xlabel('Number of Adaptation Steps')
    ax.set_ylabel('Frequency')
    ax.set_title('Adaptation Steps Distribution')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'phase2_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare PPO and Supervised adaptation results")
    parser.add_argument('--ppo_results', type=str, 
                        default='phase2/results/phase2_ppo_results.csv',
                        help='Path to PPO results CSV')
    parser.add_argument('--supervised_results', type=str,
                        default='phase2/results/phase2_supervised_results.csv',
                        help='Path to Supervised results CSV')
    parser.add_argument('--output_dir', type=str,
                        default='phase2/results',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results...")
    print(f"  PPO: {args.ppo_results}")
    print(f"  Supervised: {args.supervised_results}")
    
    ppo_df, sup_df = load_results(args.ppo_results, args.supervised_results)
    
    # Compute statistics
    ppo_stats = compute_statistics(ppo_df, 'PPO')
    sup_stats = compute_statistics(sup_df, 'Supervised')
    
    # Print comparison
    print_comparison(ppo_stats, sup_stats)
    
    # Create plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_comparison(ppo_df, sup_df, args.output_dir)
    
    print("\nDone! ✅")


if __name__ == "__main__":
    main()

