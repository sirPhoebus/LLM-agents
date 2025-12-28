#!/usr/bin/env python3
"""
Emergent Communication Training with Modular Architecture
Refactored into separate modules for cleaner and optimized code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import argparse
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
PBT_ENABLED = config['evolution'].get('pbt_enabled', False)
NUM_SPECIES = config['evolution'].get('num_species', 1) if PBT_ENABLED else 1
MUTATION_RATE = config['evolution'].get('mutation_rate', 0.01)

REWARD_WEIGHT = config['environment']['reward_weight']

# Phase 3 Task Config
NUM_SCOUTS = config.get('task', {}).get('num_scouts', 2)
PROXIMITY_WEIGHT = config.get('task', {}).get('proximity_weight', 1.0)

# Phase 4 Adversarial Config
ADV_CFG = config.get('adversarial')
NUM_ADVERSARIES = ADV_CFG.get('num_adversaries', 0) if ADV_CFG else 0
ADV_CURRICULUM = ADV_CFG.get('curriculum', {}) if ADV_CFG else {}

# Phase 7 Verification Config
VERI_CFG = config.get('verification', {})
VERIFY_ENABLED = VERI_CFG.get('enabled', False)
VERIFY_PROB = VERI_CFG.get('prob', 0.05)
VERIFY_PENALTY = VERI_CFG.get('penalty', 10.0)
VERIFY_PENALTY = VERI_CFG.get('penalty', 10.0)
VERIFY_BONUS = VERI_CFG.get('bonus', 2.0)
VERIFY_THRESHOLD = VERI_CFG.get('angle_threshold', 45.0)


# ============ TRAINING FUNCTIONS ============

def train(num_episodes=NUM_EPISODES, verbose=True, use_llm=False, llm_provider='local'):
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
        WORLD_DIM, LATENT_DIM, NUM_AGENTS, ACTION_DIM, device,
        num_scouts=NUM_SCOUTS,
        proximity_weight=PROXIMITY_WEIGHT
    )
    env.cohesion_factor = config['environment']['swarm']['cohesion_factor']
    env.separation_factor = config['environment']['swarm']['separation_factor']
    env.noise_factor = config['environment']['swarm']['noise_factor']
    env.action_scale = config['environment']['swarm']['action_scale']
    env.oscillation_amplitude = config['environment']['oscillatory']['amplitude']
    env.oscillation_frequencies = config['environment']['oscillatory']['frequencies']

    # Initialize Swarm
    # Initialize swarm model
    swarm = Swarm(
        OBS_DIM, MSG_DIM, VOCAB_SIZE, HIDDEN_DIM, LATENT_DIM, 
        NUM_AGENTS, ACTION_DIM, BATCH_SIZE, LEARNING_RATE,
        num_species=NUM_SPECIES,
        composition_config=config.get('composition'),
        num_scouts=NUM_SCOUTS,
        num_adversaries=NUM_ADVERSARIES,
        adversarial_cfg=ADV_CFG
    ).to(device)
    
    # Track current curriculum and exploration parameters
    current_k = K_NEIGHBORS_START
    neighbors = get_topology(NUM_AGENTS, TOPOLOGY, current_k)
    topology_mask = get_topology_mask(neighbors, NUM_AGENTS, device)
    
    # Track species returns for PBT (moving average)
    species_fitness_ma = torch.zeros(NUM_SPECIES, device=device)

    # Track metrics for logging
    recon_losses_history = []
    pred_losses_history = []
    reward_history = []

    for episode in range(num_episodes):
        # Reset hidden states and environment
        swarm.reset_hidden(device)
        obs, latent_z = env.reset(BATCH_SIZE, OBS_DIM)
        
        # Determine dynamic parameters
        tau = max(TAU_MIN, TAU_START * (ANNEAL_RATE ** (episode // ANNEAL_EVERY)))
        
        # Exploration/Stochasticity decay
        exploration_std = max(EXPLORATION_STD_MIN, EXPLORATION_STD_START * (STOCHASTIC_DECAY_RATE ** episode))
        policy_temp = max(POLICY_TEMP_MIN, POLICY_TEMP_START * (STOCHASTIC_DECAY_RATE ** episode))
        
        current_entropy_weight = ENTROPY_WEIGHT_END + (ENTROPY_WEIGHT_START - ENTROPY_WEIGHT_END) * \
                                 max(0, 1 - episode / ENTROPY_DECAY_EPISODES)
        current_comm_cost = COMM_COST_START + (COMM_COST_END - COMM_COST_START) * \
                            min(1.0, episode / COMM_COST_RAMP_EPISODES)
        
        # Adversarial Curriculum
        current_adv_weight = None
        if ADV_CURRICULUM.get('enabled'):
            start_ep = ADV_CURRICULUM.get('start_episode', 0)
            ramp_eps = ADV_CURRICULUM.get('ramp_episodes', 1)
            init_w = ADV_CURRICULUM.get('initial_weight', 0.0)
            final_w = ADV_CURRICULUM.get('final_weight', 1.0)
            
            if episode < start_ep:
                current_adv_weight = init_w
            else:
                progress = min(1.0, (episode - start_ep) / ramp_eps)
                current_adv_weight = init_w + (final_w - init_w) * progress
        if use_llm:
            utils.llm_bridge.log_step(obs, latent_z)
            
        # Init last_msgs (Silence)
        # Shape: (N, B, M, V). One-hot.
        last_msgs = torch.zeros(NUM_AGENTS, BATCH_SIZE, MSG_DIM, VOCAB_SIZE, device=device)
        # Set to symbol 0 (Silence)
        last_msgs[..., 0] = 1.0
        
        # Rollout
        episode_obs = []
        episode_msgs_hard = []
        episode_msgs_soft = []
        episode_actions = []
        episode_rewards = []
        
        # Phase 7 Penalty Buffer
        penalty_buffer = torch.zeros(NUM_AGENTS, BATCH_SIZE, device=device)
        
        # Optimize Verification Logic: Pre-calculate audit steps to avoid CPU-GPU sync
        # verify_steps: (T,) boolean check
        verify_steps = torch.rand(TIMESTEPS_PER_EPISODE) < VERIFY_PROB
        
        for t in range(TIMESTEPS_PER_EPISODE):
                
            msgs_hard, msgs_soft, actions = swarm.forward_policy(
                obs, last_msgs, topology_mask, tau, temperature=policy_temp, exploration_std=exploration_std
            )
            
            # Update last_msgs for next step
            # Detach to stop gradient backprop through time? 
            # Ideally BPTT works if we keep graph, but usually we truncation Backprop or use REINFORCE.
            # Here we use REINFORCE on policy outputs. 
            # If we want gradient flow through messages (Differentiation Communication), we keep graph.
            # But standard REINFORCE treats messages as actions.
            # Let's keep graph for "Differentiable Comm" if we used Gumbel-Softmax straight through?
            # Config has "gumbel_softmax".
            # If we detach, we break gradient.
            # For now, let's NOT detach, to allow gradients if supported?
            # But `last_msgs` is Input.
            # If we use REINFORCE, we don't need grad through inputs?
            # Actually, `msgs_hard` is discrete (Gumbel-Softmax ST).
            # So it has gradients.
            # If we pass it, we enable Multi-Agent Backprop!
            # Let's keep it attached.
            last_msgs = msgs_hard 
            
            episode_obs.append(obs)
            # Step environment
            obs_next, rewards, latent_z_next = env.step(actions, OBS_DIM)
            
            episode_msgs_hard.append(msgs_hard)
            episode_msgs_soft.append(msgs_soft)
            episode_actions.append(actions)
            episode_rewards.append(rewards.mean().item())
            
            obs = obs_next
            latent_z = latent_z_next
            
            if VERIFY_ENABLED and verify_steps[t]:
                 with torch.no_grad():
                     # --- Robust Verification in Latent Particle Space ---
                     # 1. Calculate Swarm Centroid and Desired Force
                     # env.action_proj: (N*A, L)
                     # latent_z: (B, L)
                     num_particles = env.latent_dim // 2
                     
                     # Current Particle Positions: (B, P, 2)
                     positions = latent_z.view(BATCH_SIZE, num_particles, 2)
                     centroid = positions.mean(dim=1) # (B, 2)
                     
                     # Vector to Target (B, 2)
                     dir_desired = env.target_pos - centroid
                     
                     # 2. Calculate Action -> Centroid Force Map
                     # reshape projector: (N*A, P*2) -> (N, A, P, 2)
                     # We can cache this, but it's cheap enough for stochastic checks
                     n_agents = NUM_AGENTS
                     a_dim = ACTION_DIM
                     denom = num_particles
                     
                     # (N, A, P, 2)
                     proj_reshaped = env.action_proj.view(n_agents, a_dim, num_particles, 2)
                     
                     # Map M: (N, A, 2). M[i] maps action vector to centroid force vector.
                     # Sum over P, divide by P (mean)
                     M = proj_reshaped.mean(dim=2) * env.action_scale # scaling matches dynamics
                     
                     # 3. Check Alignment for Followers
                     # actions: (N, B, A)
                     # We iterate followers to check mismatches
                     
                     misled_batch_indices = []
                     misled_agent_indices = []
                     
                     # We can vectorize this check
                     # actions permute: (B, N, A)
                     actions_batch = actions.permute(1, 0, 2)
                     
                     # Calculate Force Taken: (B, N, 2)
                     # einsum: bna, nac -> bnc (c=2)
                     force_taken = torch.einsum('bna,nac->bnc', actions_batch, M)
                     
                     # Calculate Alignment with dir_desired: (B, N)
                     # cosine sim
                     # normalize both
                     force_norm = F.normalize(force_taken, dim=-1)
                     dir_norm = F.normalize(dir_desired.unsqueeze(1), dim=-1) # (B, 1, 2)
                     
                     # dot product: (B, N)
                     alignment = (force_norm * dir_norm).sum(dim=-1)
                     
                     # Threshold check
                     # Angle > 90 deg => cos(angle) < 0
                     # VERIFY_THRESHOLD is in degrees.
                     thresh_cos = math.cos(math.radians(VERIFY_THRESHOLD))
                     
                     # Check followers only (>= NUM_SCOUTS)
                     # Create mask (B, N)
                     follower_mask = torch.zeros(BATCH_SIZE, NUM_AGENTS, device=device).bool()
                     follower_mask[:, NUM_SCOUTS:] = True
                     
                     # If alignment < thresh_cos, it's a "Lie" (Action contradicts Goal)
                     # For BONUS: If alignment > thresh_cos, it's "Helpful"
                     # Note: thresh_cos comes from angle_threshold (e.g. 45 deg).
                     # Lower angle = higher cos. So alignment > thresh_cos is GOOD.
                     
                     is_helpful = (alignment > thresh_cos) & follower_mask
                     is_misled = (alignment < 0) & follower_mask # Still track bad lies (opposite dir) if needed, but penalty is 0.0
                     
                     # Apply Bonus
                     if is_helpful.any():
                         attn = swarm.last_attn_weights 
                         dominant_sender_idx = attn.argmax(dim=-1) # (B, N_recv)
                         bonus_tensor = torch.tensor(VERIFY_BONUS, device=device)
                         
                         batch_idx, recv_idx = torch.where(is_helpful)
                         sender_idx = dominant_sender_idx[batch_idx, recv_idx]
                         
                         # buffer is "penalty_buffer". We can reuse it or rename it.
                         # Since final_returns = rewards - penalty_buffer, 
                         # A bonus is a NEGATIVE penalty.
                         penalty_buffer.index_put_((sender_idx, batch_idx), -bonus_tensor, accumulate=True)
            
        # Optimization
        # Stack all timesteps: (T, N, B, ...)
        all_obs = torch.stack(episode_obs)
        all_msgs_hard = torch.stack(episode_msgs_hard)
        all_msgs_soft = torch.stack(episode_msgs_soft)
        
        # Final rewards (simplified: use last step rewards for REINFORCE)
        # rewards: (B,) -> broadcast to (N, B)
        final_returns = rewards * REWARD_WEIGHT
        
        # Apply verification penalties
        # print(f"DEBUG: final_returns={final_returns.shape}, penalty_buffer={penalty_buffer.shape}")
        final_returns = final_returns.unsqueeze(0) - penalty_buffer
        
        # Compute loss and backward
        loss, recon_loss, pred_loss = swarm.compute_loss(
            all_obs.mean(0), # Average over timesteps for recon/pred
            all_msgs_hard.mean(0),
            all_msgs_soft.mean(0),
            final_returns,
            latent_z,
            topology_mask,
            current_entropy_weight,
            current_comm_cost,
            adversarial_weight_override=current_adv_weight
        )
        
        swarm.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(swarm.parameters(), MAX_GRAD_NORM)
        swarm.optimizer.step()
        swarm.scheduler.step(episode)
        
        # --- PBT Fitness Update ---
        # Calculate mean return per species across batch
        # final_returns: (B,) currently. We need to scale by individual agent contribution?
        # For now, swarm reward is shared, so all agents get same base 'final_returns'.
        # However, we can track reconstruction error per agent as a proxy for 'communication fitness'.
        
        with torch.no_grad():
            # (N, B, O)
            recon, _ = swarm.decoder(all_msgs_hard.mean(0), swarm.species_indices, topology_mask)
            # MSE per agent: (N, B)
            agent_recon_err = torch.mean((recon - all_obs.mean(0))**2, dim=-1)
            # Reward: negative error + global return
            agent_fitness = -agent_recon_err.mean(dim=1) + final_returns.mean()
            
            for s in range(NUM_SPECIES):
                mask = (swarm.species_indices == s)
                if mask.any():
                    s_fit = agent_fitness[mask].mean()
                    species_fitness_ma[s] = 0.9 * species_fitness_ma[s] + 0.1 * s_fit

        # --- PBT Selection Phase ---
        if PBT_ENABLED and episode % SELECTION_INTERVAL == 0 and episode > 0:
            print(f"\n--- [PBT] Selection Phase (Episode {episode}) ---")
            print(f"  Species Fitness: {species_fitness_ma.cpu().numpy()}")
            
            # Rank species
            sorted_idx = torch.argsort(species_fitness_ma, descending=True)
            elites = sorted_idx[:NUM_ELITE]
            to_replace = sorted_idx[-NUM_REPLACE:]
            
            for target in to_replace:
                source = elites[torch.randint(0, len(elites), (1,)).item()]
                swarm.mutate_species(target, source, MUTATION_RATE)
            print("-" * 40)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            
            recon_losses_history.append(recon_loss)
            pred_losses_history.append(pred_loss)
            reward_history.append(avg_reward)
            # Log task success (centroid distance to target)
            with torch.no_grad():
                # Centroid accurately comes from env.last_z
                positions = env.last_z.view(-1, env.num_particles, 2)
                swarm_centroid = positions.mean(dim=1) # (B, 2)
                dist = torch.norm(swarm_centroid - env.target_pos, dim=-1).mean().item()
            avg_recon = recon_loss
            avg_pred = pred_loss
            avg_reward = final_returns.mean().item()
            
            logger.log_episode(episode, recon_loss, pred_loss, avg_reward, tau, goal_dist=dist)
            
            if verbose:
                adv_str = f", AdvW={current_adv_weight:.2f}" if current_adv_weight is not None else ""
                print(f"Episode {episode}: Recon={avg_recon:.4f}, Pred={avg_pred:.4f}, Reward={avg_reward:.4f}, GoalDist={dist:.4f}, Tau={tau:.4f}, K={current_k}{adv_str}")
        
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
            
            # Trigger LLM interpretation periodically if enabled
            if use_llm:
                try:
                    from llm_bridge import run_llm_interpretation
                    run_llm_interpretation(swarm, neighbors, tau, device, env, provider=llm_provider, verbose=True)
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
        # Init last_msgs (Silence)
        last_msgs = torch.zeros(NUM_AGENTS, BATCH_SIZE, MSG_DIM, VOCAB_SIZE, device=device)
        last_msgs[..., 0] = 1.0
        
        # Policy forward
        msgs_hard, msgs_soft, actions = swarm.forward_policy(obs, last_msgs, topology_mask, tau, temperature=1.0, exploration_std=0.0)
        
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
    parser.add_argument('--use-llm', action='store_true', help='Enable periodic LLM evaluation during training')
    parser.add_argument('--analyze', type=str, help='Analyze existing log file instead of training')
    args = parser.parse_args()
    
    # If just analyzing an existing log
    if args.analyze:
        analyze_run(args.analyze)
        exit(0)
    
    # Determine LLM provider
    llm_provider = "hub" if args.hub else "local"
    
    # Run training
    swarm, final_tau, neighbors, logger, env = train(use_llm=args.use_llm, llm_provider=llm_provider)
    
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
