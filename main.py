#!/usr/bin/env python3
"""
Emergent Communication Training with Modular Architecture
Refactored into separate modules for cleaner and optimized code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Import from our new modules
from logger import TrainingLogger, analyze_run
from topology import get_topology, get_current_k, get_topology_mask
from utils import compute_returns, compute_grammar_metrics, load_config
from models import Swarm
from environment import WorldSimulator

# ============ CONFIGURATION ============

config = load_config("config.yaml")

# Device configuration
device_cfg = config['training'].get('device', 'auto')
if device_cfg == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(device_cfg)
print(f"Using device: {device}")

# Unpack config for easier access (maintaining some original variable names for compatibility)
NUM_AGENTS = config['agents']['num_agents']
WORLD_DIM = config['environment']['world_dim']
OBS_DIM = config['agents']['obs_dim']
MSG_DIM = config['agents']['msg_dim']
HIDDEN_DIM = config['agents']['hidden_dim']
LATENT_DIM = config['agents']['latent_dim']
VOCAB_SIZE = config['agents']['vocab_size']
ACTION_DIM = config['agents']['action_dim']

NUM_EPISODES = config['training']['num_episodes']
LEARNING_RATE = config['training']['learning_rate']
MAX_GRAD_NORM = config['training']['max_grad_norm']
BATCH_SIZE = config['training']['batch_size']
TIMESTEPS_PER_EPISODE = config['training']['timesteps_per_episode']
EVAL_INTERVAL = config['training']['eval_interval']

TAU_START = config['curriculum']['tau_start']
TAU_MIN = config['curriculum']['tau_min']
ANNEAL_RATE = config['curriculum']['anneal_rate']
ANNEAL_EVERY = config['curriculum']['anneal_every']

ENTROPY_WEIGHT_START = config['curriculum']['entropy_weight_start']
ENTROPY_WEIGHT_END = config['curriculum']['entropy_weight_end']
ENTROPY_DECAY_EPISODES = config['curriculum']['entropy_decay_episodes']

COMM_COST_START = config['curriculum']['comm_cost_start']
COMM_COST_END = config['curriculum']['comm_cost_end']
COMM_COST_RAMP_EPISODES = config['curriculum']['comm_cost_ramp_episodes']

EXPLORATION_STD_START = config['curriculum']['exploration_std_start']
EXPLORATION_STD_MIN = config['curriculum']['exploration_std_min']
POLICY_TEMP_START = config['curriculum']['policy_temp_start']
POLICY_TEMP_MIN = config['curriculum']['policy_temp_min']
STOCHASTIC_DECAY_RATE = config['curriculum']['stochastic_decay_rate']

TOPOLOGY = config['topology']['type']
K_NEIGHBORS_START = config['topology']['k_neighbors_start']
K_NEIGHBORS_END = config['topology']['k_neighbors_end']
K_DECAY_EPISODES = config['topology']['k_decay_episodes']

SELECTION_INTERVAL = config['evolution']['selection_interval']
NUM_ELITE = config['evolution']['num_elite']
NUM_REPLACE = config['evolution']['num_replace']

REWARD_WEIGHT = config['environment']['reward_weight']


# ============ TRAINING FUNCTIONS ============

def train(num_episodes=NUM_EPISODES, verbose=True):
    """Main training loop with vectorized Swarm architecture."""
    # Initialize logger
    logger = TrainingLogger()
    logger.log_hyperparameters(
        num_agents=NUM_AGENTS, world_dim=WORLD_DIM, obs_dim=OBS_DIM,
        msg_dim=MSG_DIM, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        entropy_start=ENTROPY_WEIGHT_START, entropy_decay=ENTROPY_DECAY_EPISODES,
        comm_cost_end=COMM_COST_END, comm_cost_ramp=COMM_COST_RAMP_EPISODES,
        timesteps=TIMESTEPS_PER_EPISODE,
        topology=TOPOLOGY, k_start=K_NEIGHBORS_START, k_end=K_NEIGHBORS_END,
        selection_interval=SELECTION_INTERVAL
    )
    
    # Initialize environment simulator
    env = WorldSimulator(
        WORLD_DIM, LATENT_DIM, NUM_AGENTS, ACTION_DIM, device
    )
    env.cohesion_factor = config['environment']['swarm']['cohesion_factor']
    env.separation_factor = config['environment']['swarm']['separation_factor']
    env.noise_factor = config['environment']['swarm']['noise_factor']
    env.action_scale = config['environment']['swarm']['action_scale']
    env.oscillation_amplitude = config['environment']['oscillatory']['amplitude']
    env.oscillation_frequencies = config['environment']['oscillatory']['frequencies']

    # Initialize Swarm
    swarm = Swarm(
        obs_dim=OBS_DIM,
        msg_dim=MSG_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        total_agents=NUM_AGENTS,
        action_dim=ACTION_DIM,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    ).to(device)

    # Topology
    current_k = K_NEIGHBORS_START
    neighbors = get_topology(NUM_AGENTS, TOPOLOGY, current_k)
    topology_mask = get_topology_mask(neighbors, NUM_AGENTS, device)
    
    tau = TAU_START
    exploration_std = EXPLORATION_STD_START
    policy_temp = POLICY_TEMP_START
    
    # Track metrics for logging
    recon_losses_history = []
    pred_losses_history = []
    reward_history = []

    for episode in range(num_episodes):
        swarm.reset_hidden(device)
        
        episode_rewards = []
        episode_data = [] # (obs, msgs_hard, msgs_soft, returns_at_t, z_tp1)
        
        # Calculate curriculum weights
        e_progress = min(1.0, episode / ENTROPY_DECAY_EPISODES)
        entropy_weight = ENTROPY_WEIGHT_START - (ENTROPY_WEIGHT_START - ENTROPY_WEIGHT_END) * e_progress
        c_progress = min(1.0, episode / COMM_COST_RAMP_EPISODES)
        comm_cost_weight = COMM_COST_START + (COMM_COST_END - COMM_COST_START) * c_progress
        
        z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        
        # 1. ROLLOUT PHASE
        for t in range(TIMESTEPS_PER_EPISODE):
            current_world, _, _, _, _ = env.generate_world_batch(BATCH_SIZE, None, z_t)
            # Obs: (N, B, obs_dim)
            obs = torch.stack([env.get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)])
            
            # Policy forward: (msgs_hard, msgs_soft, actions)
            msgs_hard, msgs_soft, actions = swarm.forward_policy(obs, tau, policy_temp, exploration_std)
            
            # Environment step
            aggregated_actions = actions.transpose(0, 1).reshape(BATCH_SIZE, -1) # (B, N*A)
            _, _, _, z_tp1, reward = env.generate_world_batch(BATCH_SIZE, aggregated_actions, z_t)
            
            episode_rewards.append(reward.mean().item())
            episode_data.append((obs, msgs_hard, msgs_soft, z_tp1, reward))
            z_t = z_tp1.detach()

        # 2. OPTIMIZATION PHASE
        returns = compute_returns(episode_rewards, device=device)
        
        total_loss = 0
        avg_recon = 0
        avg_pred = 0
        
        for t in range(TIMESTEPS_PER_EPISODE):
            obs, msgs_hard, msgs_soft, z_tp1, _ = episode_data[t]
            loss, r_loss, p_loss = swarm.compute_loss(
                obs, msgs_hard, msgs_soft, returns[t], z_tp1, topology_mask,
                entropy_weight, comm_cost_weight
            )
            total_loss += loss
            avg_recon += r_loss
            avg_pred += p_loss
            
        swarm.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(swarm.parameters(), MAX_GRAD_NORM)
        swarm.optimizer.step()
        
        # Anneal temperature and exploration
        if episode % ANNEAL_EVERY == 0:
            tau = max(TAU_MIN, tau * ANNEAL_RATE)
            exploration_std = max(EXPLORATION_STD_MIN, exploration_std * STOCHASTIC_DECAY_RATE)
            policy_temp = max(POLICY_TEMP_MIN, policy_temp * STOCHASTIC_DECAY_RATE)
        
        swarm.scheduler.step()
        
        # Log progress
        if episode % 10 == 0:
            avg_recon /= TIMESTEPS_PER_EPISODE
            avg_pred /= TIMESTEPS_PER_EPISODE
            avg_reward = np.mean(episode_rewards)
            
            recon_losses_history.append(avg_recon)
            pred_losses_history.append(avg_pred)
            reward_history.append(avg_reward)
            
            logger.log_episode(episode, avg_recon, avg_pred, avg_reward, tau)
            
            if verbose:
                print(f"Episode {episode}: Recon={avg_recon:.4f}, Pred={avg_pred:.4f}, Reward={avg_reward:.4f}, Tau={tau:.4f}, K={current_k}")
        
        # Curriculum topology
        if episode > 0 and episode % 100 == 0:
            new_k = get_current_k(episode, K_NEIGHBORS_START, K_NEIGHBORS_END, K_DECAY_EPISODES)
            if new_k != current_k:
                current_k = new_k
                neighbors = get_topology(NUM_AGENTS, TOPOLOGY, current_k)
                topology_mask = get_topology_mask(neighbors, NUM_AGENTS, device)
                if verbose:
                    print(f"  [Curriculum] K decreased to {current_k}")
        
        # Periodic evaluation every EVAL_INTERVAL episodes to monitor emergence live
        if episode > 0 and episode % EVAL_INTERVAL == 0:
            print(f"\n{'='*50}")
            print(f"PERIODIC EVALUATION (Episode {episode})")
            print(f"{'='*50}")
            eval_result = evaluate(swarm, tau, neighbors, env, verbose=True)
            
            # Log evaluation
            logger.log_evaluation(episode, {
                "vocab_used": eval_result.get("vocab_used", 0),
                "top_symbols": eval_result.get("top_symbols", []),
                "hub_counts": eval_result.get("hub_counts", {}),
                "recon_loss": eval_result.get("recon_loss", 0),
                "pred_loss": eval_result.get("pred_loss", 0),
                "current_k": current_k
            })
            
            # Trigger LLM interpretation periodically
            try:
                from llm_bridge import run_llm_interpretation
                # Determine LLM provider from argparse args if we can access them, or default from config
                # Since we are inside train(), we might not have 'args'. 
                # We can check global config or assume 'local' for now, but let's try to be smart.
                provider = config.get('llm', {}).get('provider', 'local')
                run_llm_interpretation(swarm, neighbors, tau, device, env, provider=provider, verbose=True)
            except Exception as e:
                if verbose:
                    print(f"  [LLM Bridge] Skipped: {e}")

    # Final logs
    logger.log_final({
        "final_recon": recon_losses_history[-1] if recon_losses_history else 0,
        "final_pred": pred_losses_history[-1] if pred_losses_history else 0,
        "final_reward": reward_history[-1] if reward_history else 0,
        "total_episodes": num_episodes
    })
    logger.save()

    if verbose:
        print("Training complete.")
    return swarm, tau, neighbors, logger, env


def evaluate(swarm, tau, neighbors, env, verbose=True):
    """Evaluate trained swarm and compute emergence metrics."""
    with torch.no_grad():
        # Reset hidden states
        swarm.reset_hidden(device)
        
        # Collect diagnostic data
        z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        topology_mask = get_topology_mask(neighbors, NUM_AGENTS, device)
        
        current_world, _, _, z_tp1, _ = env.generate_world_batch(BATCH_SIZE, None, z_t)
        obs = torch.stack([env.get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)])
        
        # Policy forward
        msgs_hard, msgs_soft, actions = swarm.forward_policy(obs, tau, temperature=1.0, exploration_std=0.0)
        
        # Compute losses
        loss, recon_loss, pred_loss = swarm.compute_loss(
            obs, msgs_hard, msgs_soft, torch.zeros(BATCH_SIZE, device=device), z_tp1, topology_mask, 
            0.0, 0.0
        )
        
        # Analyze symbol usage
        symbols = msgs_hard.argmax(dim=-1).cpu().numpy() # (N, B, msg_dim)
        import collections
        all_symbols = symbols.ravel()
        freq = collections.Counter(all_symbols)
        unique_symbols = list(freq.keys())
        top_symbols = freq.most_common(5)
        
        # Analyze hub counts
        hub_counts = {}
        if swarm.last_attn_weights is not None:
            # (B, N, N)
            avg_weights = swarm.last_attn_weights.mean(dim=0).cpu().numpy()
            temp_hubs = collections.Counter()
            for i in range(NUM_AGENTS):
                hub_idx = int(np.argmax(avg_weights[i]))
                if avg_weights[i, hub_idx] > 0.2:
                    temp_hubs[hub_idx] += 1
            hub_counts = dict(temp_hubs)

        if verbose:
            print(f"Avg Reconstruction Loss: {recon_loss:.4f}")
            print(f"Avg Prediction Loss: {pred_loss:.4f}")
            print(f"Used {len(unique_symbols)}/{VOCAB_SIZE} symbols")
            print(f"Top Symbols: {top_symbols}")
            if hub_counts:
                print(f"Hub Agents: {hub_counts}")

        return {
            'recon_loss': recon_loss,
            'pred_loss': pred_loss,
            'vocab_used': len(unique_symbols),
            'top_symbols': top_symbols,
            'hub_counts': hub_counts
        }


def save_swarm(swarm, prefix="swarm"):
    """Save swarm model to disk."""
    torch.save(swarm.state_dict(), f"{prefix}.pth")
    print(f"Swarm model saved as {prefix}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emergent Communication Training')
    parser.add_argument('--local', action='store_true', help='Use local LLM (localhost:1234)')
    parser.add_argument('--hub', action='store_true', help='Use HuggingFace Hub (Qwen2.5-3B)')
    parser.add_argument('--analyze', type=str, help='Analyze existing log file instead of training')
    args = parser.parse_args()
    
    # If just analyzing an existing log
    if args.analyze:
        analyze_run(args.analyze)
        exit(0)
    
    # Determine LLM provider
    llm_provider = "hub" if args.hub else "local"
    
    # Run training
    swarm, final_tau, neighbors, logger, env = train()
    
    # Final evaluation
    evaluate(swarm, final_tau, neighbors, env)
    
    # Save agents
    save_swarm(swarm)
    
    # LLM Interpretation (optional)
    try:
        from llm_bridge import run_llm_interpretation
        print(f"\nAttempting LLM interpretation (provider: {llm_provider})...")
        llm_result = run_llm_interpretation(swarm, neighbors, final_tau, device, env, provider=llm_provider)
        if llm_result:
            print("\nLLM successfully interpreted emergent patterns!")
    except ImportError:
        print("LLM bridge not available (llm_bridge.py missing)")
    except Exception as e:
        print(f"LLM interpretation skipped: {e}")
    
    # Analyze the run
    print("\n" + "="*60)
    analyze_run()
    
    print("Done.")
