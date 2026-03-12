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

def plot_algorithm_comparison(task_name="Stabilize"):
    algorithms = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]
    colors = {
        "MC": "red",
        "SARSA": "green",
        "Q_Learning": "blue",
        "Double_Q_Learning": "purple"
    }
    
    window_size = 300 
    
    # ==========================================================
    # 1. PLOT FULL-SIZE REWARDS COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        metrics_dir = os.path.join("metrics", task_name, algo)
        rewards_path = os.path.join(metrics_dir, "rewards.npy")
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(rewards_path):
            print(f"[WARNING] Skipping {algo}: Data not found at {metrics_dir}")
            continue
            
        rewards = np.load(rewards_path)
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(rewards) > window_size:
            smoothed_rewards = moving_average(rewards, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_rewards):]
            plt.plot(x_axis_smooth, smoothed_rewards, color=colors[algo], linewidth=2.5, label=algo)

    plt.title("Algorithm Comparison: Cumulative Rewards")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    reward_save_path = os.path.join("metrics", task_name, "Comparison_Rewards.png")
    plt.savefig(reward_save_path, dpi=300)
    print(f"[INFO] Saved Rewards Comparison to: {reward_save_path}")
    plt.show()

    # ==========================================================
    # 2. PLOT FULL-SIZE EPISODE LENGTH COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        metrics_dir = os.path.join("metrics", task_name, algo)
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(lengths_path):
            continue
            
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(lengths) > window_size:
            smoothed_lengths = moving_average(lengths, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_lengths):]
            plt.plot(x_axis_smooth, smoothed_lengths, color=colors[algo], linewidth=2.5, label=algo)

    plt.title("Algorithm Comparison: Episode Lengths")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Episode Length / Steps Survived (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    length_save_path = os.path.join("metrics", task_name, "Comparison_Episode_Lengths.png")
    plt.savefig(length_save_path, dpi=300)
    print(f"[INFO] Saved Episode Lengths Comparison to: {length_save_path}")
    plt.show()

if __name__ == "__main__":
    plot_algorithm_comparison(task_name="Stabilize")