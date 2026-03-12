import json
import ast
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_q_value_surface_max(q_table_path):
    print(f"[INFO] Loading Q-table from: {q_table_path}")
    
    try:
        with open(q_table_path, 'r') as f:
            data = json.load(f)
        q_values = data['q_values']  # ← fix
    except FileNotFoundError:
        print(f"[ERROR] Could not find {q_table_path}. Check the path!")
        return

    parsed_data = {}
    
    print("[INFO] Processing 4D State Space (Using Max Projection Method)...")
    for state_str, q_vals in q_values.items():
        try:
            # Convert string tuple to python tuple
            state_tuple = ast.literal_eval(state_str)
            cart_pos = state_tuple[0]
            pole_angle = state_tuple[1]
            
            # -> THE MAX PROJECTION <-
            # We ignore velocity completely and just find the highest Q-value 
            # for this specific Position + Angle combination!
            max_q = np.max(q_vals)
            
            if (cart_pos, pole_angle) not in parsed_data:
                parsed_data[(cart_pos, pole_angle)] = max_q
            else:
                parsed_data[(cart_pos, pole_angle)] = max(parsed_data[(cart_pos, pole_angle)], max_q)
                
        except Exception as e:
            continue

    if not parsed_data:
        print("[ERROR] No valid data found in Q-table.")
        return

    # Find boundaries
    positions = [k[0] for k in parsed_data.keys()]
    angles = [k[1] for k in parsed_data.keys()]
    
    min_pos, max_pos = min(positions), max(positions)
    min_ang, max_ang = min(angles), max(angles)
    
    print(f"[INFO] Found {len(parsed_data)} unique (Position, Angle) coordinates!")
    print(f"[INFO] Grid Boundaries - Cart Pos: [{min_pos}, {max_pos}], Pole Angle: [{min_ang}, {max_ang}]")

    # Create the Meshgrid
    pos_range = np.arange(min_pos, max_pos + 1)
    ang_range = np.arange(min_ang, max_ang + 1)
    Pos, Ang = np.meshgrid(pos_range, ang_range)
    
    # Fill unvisited states with the minimum known Q-value to keep the floor flat
    global_min_q = min(parsed_data.values())
    Z = np.full(Pos.shape, global_min_q, dtype=float)

    for i in range(Pos.shape[0]):
        for j in range(Pos.shape[1]):
            cp = Pos[i, j]
            pa = Ang[i, j]
            if (cp, pa) in parsed_data:
                Z[i, j] = parsed_data[(cp, pa)]

    Z = gaussian_filter(Z, sigma=1.5)

    # Plot the 3D Surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(Pos, Ang, Z, cmap='plasma', edgecolor='none', antialiased=True, alpha=0.9)
    
    ax.set_title('Policy Visualization: Maximum Q-Value Surface', fontsize=16, pad=20)
    ax.set_xlabel('\nCart Position', fontsize=12)
    ax.set_ylabel('\nPole Angle', fontsize=12)
    ax.set_zlabel('\nMax Q-Value', fontsize=12)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Max Q-Value")
    
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Q_TABLE_FILE = "q_value/Stabilize/Q_Learning/Q_Learning_2000000_15_5.0_5_25.json"
    plot_q_value_surface_max(Q_TABLE_FILE)