import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(a, n=100):
    """Calculates a simple moving average to smooth out noisy RL data."""
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_resolution_path(task_name, res, default_algo="Q_Learning"):
    """Clean routing strictly for EX2 State Discretization."""
    return os.path.join("metrics", task_name, "EX2", res)

def plot_resolution_comparison(task_name="Stabilize"):
    resolutions = ["coarse", "medium", "fine"]
    colors = {
        "coarse": "red",
        "medium": "green",
        "fine": "blue",
    }
    
    window_size = 2000 
    
    # ==========================================================
    # 1. PLOT FULL-SIZE REWARDS COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    for res in resolutions:
        metrics_dir = get_resolution_path(task_name, res)
        rewards_path = os.path.join(metrics_dir, "rewards.npy")
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(rewards_path):
            print(f"[WARNING] Skipping '{res}': Data not found at {metrics_dir}")
            continue
            
        rewards = np.load(rewards_path)
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(rewards) > window_size:
            smoothed_rewards = moving_average(rewards, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_rewards):]
            plt.plot(x_axis_smooth, smoothed_rewards, color=colors[res], linewidth=2.5, label=res.capitalize())

    plt.title("State Discretization (EX2): Cumulative Rewards")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    reward_save_path = os.path.join("metrics", task_name, "EX2_Comparison_Rewards.png")
    os.makedirs(os.path.dirname(reward_save_path), exist_ok=True)
    plt.savefig(reward_save_path, dpi=300)
    print(f"[INFO] Saved EX2 Rewards Comparison to: {reward_save_path}")
    plt.show()

    # ==========================================================
    # 2. PLOT FULL-SIZE EPISODE LENGTH COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    for res in resolutions:
        metrics_dir = get_resolution_path(task_name, res)
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(lengths_path):
            continue
            
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(lengths) > window_size:
            smoothed_lengths = moving_average(lengths, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_lengths):]
            plt.plot(x_axis_smooth, smoothed_lengths, color=colors[res], linewidth=2.5, label=res.capitalize())

    plt.title("State Discretization (EX2): Episode Lengths")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Steps Survived (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    length_save_path = os.path.join("metrics", task_name, "EX2_Comparison_Lengths.png")
    plt.savefig(length_save_path, dpi=300)
    print(f"[INFO] Saved EX2 Episode Lengths Comparison to: {length_save_path}")
    plt.show()

if __name__ == "__main__":
    plot_resolution_comparison(task_name="Stabilize")