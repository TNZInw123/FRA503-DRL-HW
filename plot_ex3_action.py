import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(a, n=100):
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_action_path(task_name, act_val):
    """Clean routing strictly for EX3 Action Spaces."""
    return os.path.join("metrics", task_name, "EX3", f"act{act_val}")

def plot_action_comparison(task_name, actions):
    # Hardcoded color progression to match EX1 and EX2 perfectly
    base_colors = ["red", "green", "blue"]
    colors = {act: base_colors[i] for i, act in enumerate(actions)}
    
    # CRANKED UP to 2000 so your EX3 lines are buttery smooth!
    window_size = 2000 
    
    # ==========================================================
    # 1. PLOT FULL-SIZE REWARDS COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for act in actions:
        metrics_dir = get_action_path(task_name, act)
        rewards_path = os.path.join(metrics_dir, "rewards.npy")
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(rewards_path):
            print(f"[WARNING] Skipping Act {act}: Data not found at {metrics_dir}")
            continue
            
        rewards = np.load(rewards_path)
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(rewards) > window_size:
            smoothed_rewards = moving_average(rewards, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_rewards):]
            plt.plot(x_axis_smooth, smoothed_rewards, color=colors[act], linewidth=2.5, label=f"{act} Actions")

    plt.title("Action Space (EX3): Cumulative Rewards")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    reward_save_path = os.path.join("metrics", task_name, "EX3_Comparison_Rewards.png")
    os.makedirs(os.path.dirname(reward_save_path), exist_ok=True)
    plt.savefig(reward_save_path, dpi=300)
    print(f"[INFO] Saved EX3 Rewards Comparison to: {reward_save_path}")
    plt.show()

    # ==========================================================
    # 2. PLOT FULL-SIZE EPISODE LENGTH COMPARISON
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for act in actions:
        metrics_dir = get_action_path(task_name, act)
        lengths_path = os.path.join(metrics_dir, "lengths.npy")
        
        if not os.path.exists(lengths_path):
            continue
            
        lengths = np.load(lengths_path)
        cumulative_steps = np.cumsum(lengths)
        
        if len(lengths) > window_size:
            smoothed_lengths = moving_average(lengths, n=window_size)
            x_axis_smooth = cumulative_steps[len(cumulative_steps) - len(smoothed_lengths):]
            plt.plot(x_axis_smooth, smoothed_lengths, color=colors[act], linewidth=2.5, label=f"{act} Actions")

    plt.title("Action Space (EX3): Episode Lengths")
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Steps Survived (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    length_save_path = os.path.join("metrics", task_name, "EX3_Comparison_Lengths.png")
    plt.savefig(length_save_path, dpi=300)
    print(f"[INFO] Saved EX3 Episode Lengths Comparison to: {length_save_path}")
    plt.show()

if __name__ == "__main__":
    # List the exactly 3 action counts you are testing in EX3
    ACTION_SPACES = [2, 9, 15] 
    
    plot_action_comparison(task_name="Stabilize", actions=ACTION_SPACES)