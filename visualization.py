import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import train, generate_world_batch, get_partial_obs, NUM_AGENTS, OBS_DIM, BATCH_SIZE, WORLD_DIM, VOCAB_SIZE, NUM_PARTICLES

def visualize_emergence():
    print("Training agents...")
    agents, tau = train()  # Assuming train now returns agents, tau
    
    print("Generating validation data...")
    # Generate a trajectory for visualization
    traj_length = 50
    z_true_history = []
    z_pred_history = []  # List of [agent_preds] per step, each agent_pred is (1, LATENT_DIM)
    
    # Per-step symbols: list of [agent_symbols] per step, each (1, MSG_DIM)
    symbols_history = []
    
    from main import apply_swarm_dynamics, PROJECTION_MAT, MSG_DIM
    
    z_curr = torch.randn(1, 8)  # LATENT_DIM=8
    
    preds_per_agent = [[] for _ in range(NUM_AGENTS)]  # Will be lists of arrays
    
    for step in range(traj_length):
        z_true_history.append(z_curr.detach().numpy())
        
        # Project -> Observe
        current_world = z_curr @ PROJECTION_MAT
        obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
        
        # Agents generate messages
        msgs = [agents[i].generate_message(obs_list[i], tau) for i in range(NUM_AGENTS)]
        symbols = [msg.argmax(dim=-1).cpu().numpy() for msg in msgs]  # (NUM_AGENTS, (1, MSG_DIM))
        symbols_history.append(symbols)
        
        # Agents predict global state (actually, estimate hat_z, but for traj viz, we use hat_z_next? Wait, user asked for pred traj, but since it's 1-step, we'll use hat_z_next as pred for next z)
        agent_preds_step = []
        for i in range(NUM_AGENTS):
            received = [msgs[j] for j in range(NUM_AGENTS) if j != i]
            all_msgs = [msgs[i]] + received
            all_concat = torch.cat(all_msgs, dim=1).view(1, -1)
            hat_z = agents[i].world_estimator(all_concat)
            hat_z_next = agents[i].transition(hat_z)  # Use predicted next as the "pred" for traj
            agent_preds_step.append(hat_z_next.detach().numpy())
        
        preds_per_agent = [p + [ap] for p, ap in zip(preds_per_agent, agent_preds_step)]  # Append
        
        # Step Physics
        z_curr = apply_swarm_dynamics(z_curr)

    z_true_history = np.array(z_true_history).squeeze()  # (T, 8)
    preds_per_agent = np.array([np.array(p).squeeze() for p in preds_per_agent])  # (A, T, 8)
    
    # NEW: Multi-particle trajectory plots
    colors = ['r', 'g', 'b']
    fig_multi, axs_multi = plt.subplots(2, 2, figsize=(12, 12))
    axs_multi = axs_multi.ravel()
    for particle in range(NUM_PARTICLES):
        ax = axs_multi[particle]
        # Plot true traj for this particle (dims 2*particle : 2*particle+2 for x,y)
        x_dim, y_dim = 2 * particle, 2 * particle + 1
        ax.plot(z_true_history[:, x_dim], z_true_history[:, y_dim], 'k-', linewidth=2, label='Ground Truth')
        for i in range(NUM_AGENTS):
            ax.plot(preds_per_agent[i, :, x_dim], preds_per_agent[i, :, y_dim], f'{colors[i]}--', label=f'Agent {i} Pred')
        ax.set_title(f"Swarm Particle {particle} Trajectory (1-Step Prediction)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
    plt.tight_layout()
    plt.savefig('c:/repo/agent/LLM-agents/multi_particle_traj.png')
    print("Multi-particle trajectory saved to c:/repo/agent/LLM-agents/multi_particle_traj.png")
    
    # NEW: Time-series error plot: MSE between true z and avg agent pred over steps
    avg_preds = np.mean(preds_per_agent, axis=0)  # (T, 8)
    mse_per_step = np.mean((z_true_history - avg_preds)**2, axis=1)  # (T,)
    plt.figure(figsize=(10, 5))
    plt.plot(range(traj_length), mse_per_step, 'b-', label='MSE')
    plt.title("Time-Series MSE: True Z vs Avg Agent Pred")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig('c:/repo/agent/LLM-agents/mse_timeseries.png')
    print("MSE time-series saved to c:/repo/agent/LLM-agents/mse_timeseries.png")
    
    # NEW: Symbol activation tied to latent values
    # For simplicity, bin each latent dim into 5 bins, and count symbol freq per bin per agent (aggregate over steps and msg dims)
    # We'll make a heatmap per agent per latent dim, but to keep it manageable, perhaps one big heatmap: agents x symbols, but colored by bin? Wait, better: for each latent dim, a symbols x bins heatmap (avg over agents)
    num_bins = 5
    symbols_history = np.array(symbols_history)  # (T, A, 1, MSG_DIM) but squeeze to (T, A, MSG_DIM)
    symbols_history = symbols_history.squeeze(2)  # Assuming batch=1
    
    # Flatten over T, A, MSG_DIM to get all symbol activations, with corresponding z_t (repeat z_t for A*MSG_DIM)
    all_symbols = symbols_history.reshape(-1)  # (T*A*MSG_DIM,)
    all_z = np.tile(z_true_history, (1, NUM_AGENTS * MSG_DIM)).reshape(-1, LATENT_DIM)  # (T*A*MSG_DIM, LATENT_DIM)
    
    for dim in range(LATENT_DIM):
        # Bin the z values for this dim
        z_vals = all_z[:, dim]
        bins = np.linspace(z_vals.min(), z_vals.max(), num_bins + 1)
        bin_indices = np.digitize(z_vals, bins) - 1  # 0 to num_bins-1
        
        # Count freq of each symbol in each bin
        freq_matrix = np.zeros((VOCAB_SIZE, num_bins))
        for s, b in zip(all_symbols, bin_indices):
            freq_matrix[int(s), int(b)] += 1
        
        # Normalize rows (per symbol)
        row_sums = freq_matrix.sum(axis=1, keepdims=True)
        freq_matrix = np.divide(freq_matrix, row_sums, where=row_sums != 0)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(freq_matrix, cmap="viridis", cbar_kws={'label': 'Normalized Freq'})
        plt.title(f"Symbol Activation by Latent Dim {dim} Bins")
        plt.xlabel("Bin Index")
        plt.ylabel("Symbol ID")
        plt.tight_layout()
        plt.savefig(f'c:/repo/agent/LLM-agents/symbol_activation_dim_{dim}.png')
        print(f"Symbol activation for dim {dim} saved to c:/repo/agent/LLM-agents/symbol_activation_dim_{dim}.png")
    
    # Original plots (for completeness)
    # ... (keep the swarm particle 0 and symbol usage heatmap, save to emergence_viz.png)

if __name__ == "__main__":
    visualize_emergence()   