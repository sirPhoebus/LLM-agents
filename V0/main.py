import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime


# ============ TRAINING LOGGER ============

class TrainingLogger:
    """Logs training metrics and emergence signals to JSON file."""
    
    def __init__(self, log_file: str = "training_log.json"):
        self.log_file = log_file
        self.start_time = time.time()
        self.data = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "hyperparameters": {},
            "episodes": [],
            "evaluations": [],
            "final_metrics": {}
        }
    
    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters at start of run."""
        self.data["hyperparameters"] = kwargs
    
    def log_episode(self, episode: int, recon_loss: float, pred_loss: float, 
                    reward_loss: float, tau: float, vocab_used: int = None):
        """Log metrics for an episode."""
        self.data["episodes"].append({
            "episode": episode,
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "reward_loss": reward_loss,
            "tau": tau,
            "vocab_used": vocab_used,
            "elapsed_sec": time.time() - self.start_time
        })
    
    def log_evaluation(self, episode: int, eval_data: dict):
        """Log periodic evaluation results."""
        self.data["evaluations"].append({
            "episode": episode,
            **eval_data
        })
    
    def log_final(self, final_metrics: dict):
        """Log final run metrics."""
        self.data["final_metrics"] = final_metrics
        self.data["end_time"] = datetime.now().isoformat()
        self.data["total_duration_sec"] = time.time() - self.start_time
    
    def save(self):
        """Save log to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"Training log saved to {self.log_file}")
    
    def get_data(self) -> dict:
        """Get log data for analysis."""
        return self.data


def analyze_run(log_file: str = "training_log.json"):
    """
    Analyze a training run from the log file and print summary.
    """
    with open(log_file, "r") as f:
        data = json.load(f)
    
    episodes = data.get("episodes", [])
    evaluations = data.get("evaluations", [])
    hp = data.get("hyperparameters", {})
    final = data.get("final_metrics", {})
    
    print("\n" + "="*60)
    print("TRAINING RUN ANALYSIS")
    print("="*60)
    print(f"Run ID: {data.get('run_id', 'unknown')}")
    print(f"Duration: {data.get('total_duration_sec', 0)/60:.1f} minutes")
    print(f"Episodes: {len(episodes)}")
    
    # Hyperparameters
    print("\n--- Hyperparameters ---")
    for key, val in hp.items():
        print(f"  {key}: {val}")
    
    # Loss trajectory
    if episodes:
        first_10 = episodes[:10]
        last_10 = episodes[-10:]
        
        avg_recon_start = np.mean([e["recon_loss"] for e in first_10])
        avg_recon_end = np.mean([e["recon_loss"] for e in last_10])
        avg_pred_start = np.mean([e["pred_loss"] for e in first_10])
        avg_pred_end = np.mean([e["pred_loss"] for e in last_10])
        
        print("\n--- Loss Trajectory ---")
        print(f"  Recon Loss: {avg_recon_start:.4f} → {avg_recon_end:.4f} ({(1-avg_recon_end/avg_recon_start)*100:.1f}% reduction)")
        print(f"  Pred Loss:  {avg_pred_start:.4f} → {avg_pred_end:.4f} ({(1-avg_pred_end/avg_pred_start)*100:.1f}% reduction)")
    
    # Emergence signals
    print("\n--- Emergence Signals ---")
    if evaluations:
        last_eval = evaluations[-1]
        vocab_used = last_eval.get("vocab_used", "N/A")
        hub_counts = last_eval.get("hub_counts", {})
        
        print(f"  Vocab Used: {vocab_used}/64")
        if hub_counts:
            top_hub = max(hub_counts.items(), key=lambda x: x[1]) if hub_counts else None
            print(f"  Top Hub: Agent {top_hub[0]} (attended by {top_hub[1]} agents)" if top_hub else "  No clear hubs")
        
        # Check for temporal patterns
        temporal_patterns = last_eval.get("temporal_patterns", [])
        if temporal_patterns:
            print(f"  Temporal Patterns: {len(temporal_patterns)} detected")
    else:
        print("  No evaluations logged")
    
    # Final verdict
    print("\n--- Emergence Assessment ---")
    if evaluations:
        last_eval = evaluations[-1]
        vocab = last_eval.get("vocab_used", 64)
        hubs = len(last_eval.get("hub_counts", {}))
        
        emergence_score = 0
        if vocab < 45:
            emergence_score += 2
            print("  ✓ Symbol specialization (vocab < 70%)")
        if vocab < 32:
            emergence_score += 1
            print("  ✓ Strong specialization (vocab < 50%)")
        if hubs > 0:
            emergence_score += 2
            print("  ✓ Hub/relay patterns detected")
        if avg_recon_end < avg_recon_start * 0.5:
            emergence_score += 1
            print("  ✓ Significant loss reduction")
        
        print(f"\n  EMERGENCE SCORE: {emergence_score}/6")
        if emergence_score >= 4:
            print("  → Strong signs of emergence!")
        elif emergence_score >= 2:
            print("  → Weak emergence, needs more training")
        else:
            print("  → No clear emergence yet")
    
    print("="*60)
    return data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# GPU Optimizations (disabled for compatibility - enable if Triton is installed)
# if device.type == 'cuda':
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     print("CUDA optimizations enabled")

# Hyperparameters
NUM_AGENTS = 10  # Scaled up for sparse communication
WORLD_DIM = 10
OBS_DIM = 4
MSG_DIM = 2       # CRITICAL: fewer slots forces compression
HIDDEN_DIM = 64
NUM_EPISODES = 5000
LEARNING_RATE = 3e-4  # Stable for emergence
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 512  # Larger batch for stability
VOCAB_SIZE = 16   # CRITICAL: smaller vocab forces specialization
TAU_START = 1.5   # Start warmer for exploration
TAU_MIN = 0.05    # Don't go too cold
ANNEAL_RATE = 0.999
ANNEAL_EVERY = 10

# CURRICULUM: these are START values, will ramp during training
ENTROPY_WEIGHT_START = 0.01   # Explore early
ENTROPY_WEIGHT_END = 0.0      # Zero by episode 1500
ENTROPY_DECAY_EPISODES = 1500

COMM_COST_START = 0.0         # No cost early (learn what to say)
COMM_COST_END = 0.1           # Strong cost by episode 3000
COMM_COST_RAMP_EPISODES = 3000

ACTION_DIM = 4
REWARD_WEIGHT = 2.0
TIMESTEPS_PER_EPISODE = 9

# Stochasticity & Exploration Schedules
EXPLORATION_STD_START = 0.5
EXPLORATION_STD_MIN = 0.1
POLICY_TEMP_START = 1.0
POLICY_TEMP_MIN = 0.1
STOCHASTIC_DECAY_RATE = 0.995  # Every 10 episodes

# Topology settings
TOPOLOGY = 'knearest'
K_NEIGHBORS_START = 8
K_NEIGHBORS_END = 2
K_DECAY_EPISODES = 3000

# Survival Selection settings (delayed until after emergence starts)
SELECTION_INTERVAL = 500  # Less frequent, only after some convergence
NUM_ELITE = 3
NUM_REPLACE = 3

LATENT_DIM = 8
NUM_PARTICLES = LATENT_DIM // 2

PROJECTION_MAT = torch.randn(LATENT_DIM, WORLD_DIM).to(device)

COHESION_FACTOR = 0.01
SEPARATION_FACTOR = 0.05
NOISE_FACTOR = 0.05
ACTION_SCALE = 0.1  # Scale factor for applying actions to latent
ACTION_PROJ = torch.randn(NUM_AGENTS * ACTION_DIM, LATENT_DIM).to(device)  # Project concat actions to latent space


# ============ TOPOLOGY FUNCTIONS ============

def get_topology(num_agents, topology='full', k=2):
    """
    Returns adjacency list: neighbors[i] = list of agent indices that agent i receives from.
    """
    if topology == 'full':
        return {i: [j for j in range(num_agents) if j != i] for i in range(num_agents)}
    
    elif topology == 'ring':
        neighbors = {}
        for i in range(num_agents):
            left = (i - 1) % num_agents
            right = (i + 1) % num_agents
            neighbors[i] = [left, right]
        return neighbors
    
    elif topology == 'knearest':
        neighbors = {}
        for i in range(num_agents):
            nbs = []
            for offset in range(1, k + 1):
                nbs.append((i - offset) % num_agents)
                nbs.append((i + offset) % num_agents)
            neighbors[i] = sorted(set(nbs))
        return neighbors
    
    else:
        raise ValueError(f"Unknown topology: {topology}")


def get_current_k(episode, k_start=K_NEIGHBORS_START, k_end=K_NEIGHBORS_END, decay_episodes=K_DECAY_EPISODES):
    """
    Curriculum topology: k decays linearly from k_start to k_end.
    Dense at start (learn what to communicate), sparse at end (learn efficiency).
    """
    if episode >= decay_episodes:
        return k_end
    progress = episode / decay_episodes
    k = k_start - (k_start - k_end) * progress
    return max(k_end, int(k))


def survival_selection(agents, fitness_scores):
    """
    Evolutionary pressure: Clone top agents, replace bottom agents.
    
    Args:
        agents: List of Agent objects
        fitness_scores: List of fitness scores (higher is better, e.g. -loss)
    
    Returns:
        Updated agents list
    """
    import copy
    
    # Sort by fitness (higher is better)
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    
    elite_indices = sorted_indices[:NUM_ELITE]
    weak_indices = sorted_indices[-NUM_REPLACE:]
    
    # Clone elite agents to replace weak ones
    for i, weak_idx in enumerate(weak_indices):
        elite_idx = elite_indices[i % NUM_ELITE]
        # Deep copy the elite agent's state
        agents[weak_idx].load_state_dict(copy.deepcopy(agents[elite_idx].state_dict()))
        # CRITICAL: Also copy the optimizer state to avoid resetting momentum/learning
        agents[weak_idx].optimizer.load_state_dict(copy.deepcopy(agents[elite_idx].optimizer.state_dict()))
    
    return agents, elite_indices, weak_indices


def visualize_topology(neighbors, num_agents):
    fig, ax = plt.subplots(figsize=(6, 6))
    angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for i, nbs in neighbors.items():
        for j in nbs:
            ax.plot([positions[i, 0], positions[j, 0]], 
                   [positions[i, 1], positions[j, 1]], 'b-', alpha=0.3, linewidth=0.5)
    
    ax.scatter(positions[:, 0], positions[:, 1], s=200, c='orange', zorder=5)
    for i in range(num_agents):
        ax.annotate(str(i), positions[i], ha='center', va='center', fontsize=10, zorder=6)
    
    ax.set_title(f"Communication Topology: {TOPOLOGY} (N={num_agents})")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('topology.png')
    print("Topology saved to topology.png")
    return fig


# ============ GUMBEL SOFTMAX ============

def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = torch.nn.functional.softmax(gumbels, dim=-1)
    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, y_soft


# ============ NEURAL MODULES ============

class LSTMEncoder(nn.Module):
    """LSTM-based Encoder with memory across timesteps."""
    def __init__(self, input_dim, hidden_dim, msg_dim, vocab_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        
        # Project observation to LSTM input space
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output projection to message logits
        self.output_proj = nn.Linear(hidden_dim, msg_dim * vocab_size)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state for a new trajectory."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        device = x.device
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # Project and add sequence dimension
        x = torch.relu(self.input_proj(x))  # (B, hidden_dim)
        x = x.unsqueeze(1)  # (B, 1, hidden_dim) - single timestep
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (B, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (B, hidden_dim)
        
        # Project to message logits
        logits = self.output_proj(lstm_out).view(batch_size, self.msg_dim, self.vocab_size)
        
        return logits, hidden


class ActionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # Bounded actions


class Decoder(nn.Module):
    """Original concatenation-based decoder (kept for reference)."""
    def __init__(self, num_neighbors, msg_dim, vocab_size, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        input_dim = num_neighbors * msg_dim * vocab_size
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MessageAttention(nn.Module):
    """Multi-head attention over received messages with LayerNorm for stability."""
    def __init__(self, msg_dim, vocab_size, hidden_dim, num_heads=4):
        super(MessageAttention, self).__init__()
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        self.msg_flat_dim = msg_dim * vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Learned query (used when own_msg is None to prevent self-info leakage)
        self.learned_query = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Project messages to query/key/value
        self.query_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.key_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.value_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        
        # LayerNorm for stability
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_key = nn.LayerNorm(hidden_dim)
        self.norm_value = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, own_msg, received_msgs):
        num_neighbors = len(received_msgs)
        batch_size = received_msgs[0].size(0)
        
        # Stack neighbors: (B, num_neighbors, msg_flat_dim)
        neighbors_flat = torch.stack([m.view(batch_size, -1) for m in received_msgs], dim=1)
        
        # Query: use learned query if own_msg is None (prevents self-info leakage)
        if own_msg is None:
            query = self.norm_query(self.learned_query.expand(batch_size, -1)).unsqueeze(1)
        else:
            own_flat = own_msg.view(batch_size, -1)
            query = self.norm_query(self.query_proj(own_flat)).unsqueeze(1)
        
        keys = self.norm_key(self.key_proj(neighbors_flat))
        values = self.norm_value(self.value_proj(neighbors_flat))
        
        # Multi-head attention
        attn_out, attn_weights = self.attention(query, keys, values)
        attn_out = attn_out.squeeze(1)
        
        # Output projection with LayerNorm
        out = self.norm_out(self.out_proj(attn_out))
        
        return out, attn_weights.squeeze(1)


class AttentionDecoder(nn.Module):
    """Decoder using attention over neighbor messages."""
    def __init__(self, msg_dim, vocab_size, hidden_dim, output_dim, num_heads=4):
        super(AttentionDecoder, self).__init__()
        self.msg_attention = MessageAttention(msg_dim, vocab_size, hidden_dim, num_heads)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, own_msg, received_msgs):
        """
        own_msg: (B, msg_dim, vocab_size)
        received_msgs: list of (B, msg_dim, vocab_size)
        Returns: reconstructed obs (B, output_dim), attention weights (B, num_neighbors)
        """
        attended, attn_weights = self.msg_attention(own_msg, received_msgs)
        x = torch.relu(self.fc1(attended))
        out = self.fc2(x)
        return out, attn_weights


class WorldModel(nn.Module):
    def __init__(self, msg_dim, vocab_size, hidden_dim, latent_dim, num_heads=4):
        super(WorldModel, self).__init__()
        self.world_attention = MessageAttention(msg_dim, vocab_size, hidden_dim, num_heads)
        self.world_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def estimate_latent(self, received_msgs):
        if not received_msgs:
            return torch.zeros(BATCH_SIZE, LATENT_DIM, device=device)
        # Use mean of received as query
        query = torch.mean(torch.stack(received_msgs), dim=0)
        attended, _ = self.world_attention(query, received_msgs)
        return self.world_proj(attended)

    def predict_next(self, hat_z):
        return self.transition(hat_z)


class Policy(nn.Module):
    def __init__(self, obs_dim, msg_dim, vocab_size, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.encoder = LSTMEncoder(obs_dim, hidden_dim, msg_dim, vocab_size)
        self.action_head = ActionHead(obs_dim, hidden_dim, action_dim)

    def forward(self, obs, hidden_state):
        # Generates both message logits and action mean
        msg_logits, new_hidden = self.encoder(obs, hidden_state)
        action_mean = self.action_head(obs)
        return msg_logits, action_mean, new_hidden


class Agent(nn.Module):
    def __init__(self, obs_dim, msg_dim, vocab_size, hidden_dim, latent_dim, 
                 num_neighbors, total_agents, action_dim, use_lstm=True):
        super(Agent, self).__init__()
        self.vocab_size = vocab_size
        self.policy = Policy(obs_dim, msg_dim, vocab_size, hidden_dim, action_dim).to(device)
        self.world_model = WorldModel(msg_dim, vocab_size, hidden_dim, latent_dim).to(device)
        self.decoder = AttentionDecoder(msg_dim, vocab_size, hidden_dim, obs_dim).to(device) 
        self.last_attn_weights = None
        self.hidden_state = None 

        # Create optimizers AFTER moving modules to device
        self.world_model_optimizer = optim.Adam(
            list(self.world_model.parameters()) + list(self.decoder.parameters()), 
            lr=LEARNING_RATE
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

        # Schedulers
        self.world_model_scheduler = optim.lr_scheduler.LambdaLR(
            self.world_model_optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / 100.0)
        )
        self.policy_scheduler = optim.lr_scheduler.LambdaLR(
            self.policy_optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / 100.0)
        )

    def reset_hidden(self):
        self.hidden_state = None

    def generate_message(self, obs, tau, temperature=1.0):
        # msg_logits, action_mean, new_hidden
        msg_logits, _, self.hidden_state = self.policy(obs, self.hidden_state)
        # Scale logits by temperature for exploration
        scaled_logits = msg_logits / temperature
        hard, soft = gumbel_softmax(scaled_logits, tau, hard=True)
        return hard, soft

    def generate_action(self, obs, exploration_std=0.1):
        _, action_mean, _ = self.policy(obs, self.hidden_state)
        # Add Gaussian noise
        noise = torch.randn_like(action_mean) * exploration_std
        return action_mean + noise

    def reconstruct(self, received_msgs):
        recon, attn_weights = self.decoder(None, received_msgs)
        self.last_attn_weights = attn_weights
        return recon

    def estimate_latent(self, received_msgs):
        return self.world_model.estimate_latent(received_msgs)

    def predict_next(self, hat_z):
        return self.world_model.predict_next(hat_z)

    def compute_loss(self, received_msgs, true_obs, own_msg_hard, own_msg_soft, target_z, returns_trajectory, 
                     episode, entropy_weight, comm_cost_weight):
        # 1. SSL losses for world model (dense, smooth signals)
        recon = self.reconstruct(received_msgs)
        recon_loss = nn.MSELoss()(recon, true_obs)
        
        hat_z = self.estimate_latent(received_msgs)
        hat_z_next = self.predict_next(hat_z)
        pred_loss = nn.MSELoss()(hat_z_next, target_z)
        
        # Entropy and comm cost
        entropy = -torch.mean(own_msg_soft * torch.log(own_msg_soft + 1e-10))
        entropy_loss = -entropy_weight * entropy
        comm_cost = comm_cost_weight * torch.mean(torch.abs(own_msg_soft))
        
        ssl_loss = recon_loss + pred_loss + entropy_loss + comm_cost
        
        # 2. RL loss for policy: REINFORCE with returns
        # msg_logprobs: Log prob of the chosen symbol (hard message)
        # cross_entropy(soft_logits, hard_argmax) is effectively -log_prob
        msg_logprobs = -nn.CrossEntropyLoss(reduction='none')(
            own_msg_soft.view(-1, self.vocab_size), 
            own_msg_hard.argmax(-1).view(-1)
        ).view(own_msg_soft.size(0), -1).mean(dim=-1)
        
        # For simplicity, we optimize messages via REINFORCE using trajectory returns
        rl_loss = -torch.mean(msg_logprobs * returns_trajectory)
        
        return ssl_loss, rl_loss, recon_loss.item(), pred_loss.item(), returns_trajectory.mean().item()

    def state_dict(self):
        return {
            'policy': self.policy.state_dict(),
            'world_model': self.world_model.state_dict(),
            'decoder': self.decoder.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        self.world_model.load_state_dict(state_dict['world_model'])
        self.decoder.load_state_dict(state_dict['decoder'])


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns for credit assignment."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    # Normalize returns across the trajectory for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ============ WORLD SIMULATION ============

def apply_swarm_dynamics(z: torch.Tensor, aggregated_actions: torch.Tensor) -> torch.Tensor:
    positions = z.view(-1, NUM_PARTICLES, 2)
    center = positions.mean(dim=1, keepdim=True)
    cohesion = COHESION_FACTOR * (center - positions)
    
    diffs = positions.unsqueeze(2) - positions.unsqueeze(1)
    dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
    close_mask = (dists < 1.0).float()
    repulsion = SEPARATION_FACTOR * (diffs / dists) * close_mask
    separation = repulsion.sum(dim=2) / (close_mask.sum(dim=2) + 1e-6)
    
    # Apply concatenated actions projected to particle space
    # aggregated_actions: (B, NUM_AGENTS * ACTION_DIM)
    action_forces = (aggregated_actions @ ACTION_PROJ).view(-1, NUM_PARTICLES, 2) * ACTION_SCALE
    
    # Apply oscillatory forces (environment-level periodic dynamics)
    oscillatory_force = compute_oscillatory_force(positions, global_time_step[0])
    global_time_step[0] += 1  # Increment time
    
    delta = cohesion + separation + oscillatory_force + NOISE_FACTOR * torch.randn_like(positions) + action_forces
    new_positions = positions + delta
    
    # Compute reward: negative variance (encourage clustering)
    reward = -new_positions.var(dim=1).mean(dim=-1)
    
    return new_positions.view(-1, LATENT_DIM), reward


# Oscillatory World Dynamics
OSCILLATION_AMPLITUDE = 0.1  # Strength of periodic forces
OSCILLATION_FREQUENCIES = [0.05, 0.08, 0.12]  # Multiple frequencies for complexity
global_time_step = [0]  # Mutable global counter

def compute_oscillatory_force(positions, time_step):
    """
    Apply periodic forces to swarm particles.
    Agents must communicate to track/anticipate these oscillations.
    
    Uses multiple sinusoidal frequencies to create complex, non-trivial patterns
    that require coordination to predict.
    """
    batch_size, num_particles, dims = positions.shape
    
    # Multi-frequency oscillation (different particles feel different phases)
    force = torch.zeros_like(positions)
    
    for i, freq in enumerate(OSCILLATION_FREQUENCIES):
        # Phase offset per particle creates spatial variation
        phase_offset = torch.arange(num_particles, device=positions.device).float() * 0.5
        
        # X-direction force: sin wave
        t = time_step * freq
        force[:, :, 0] += OSCILLATION_AMPLITUDE * torch.sin(t + phase_offset)
        
        # Y-direction force: cos wave (90 degrees out of phase)
        force[:, :, 1] += OSCILLATION_AMPLITUDE * torch.cos(t + phase_offset * 0.7)
    
    # Normalize by number of frequencies
    force = force / len(OSCILLATION_FREQUENCIES)
    
    return force


def generate_world_batch(batch_size, world_dim, aggregated_actions, z_t=None):
    if z_t is None:
        z_t = torch.randn(batch_size, LATENT_DIM, device=device)
    current_world = z_t @ PROJECTION_MAT
    z_tp1, reward = apply_swarm_dynamics(z_t, aggregated_actions)
    target_world = z_tp1 @ PROJECTION_MAT
    return current_world, target_world, z_t, z_tp1, reward


def get_partial_obs(world, agent_id, obs_dim):
    """
    CRITICAL: Complementary observability.
    Each agent sees a different slice of the world state (wraps around).
    Communication is REQUIRED to reconstruct the full world.
    """
    start = (agent_id * obs_dim) % world.shape[1]
    indices = torch.arange(start, start + obs_dim, device=device) % world.shape[1]
    return world.index_select(1, indices)


# ============ TRAINING ============

def train(num_episodes=NUM_EPISODES, verbose=True):
    # Initialize logger
    logger = TrainingLogger()
    logger.log_hyperparameters(
        num_agents=NUM_AGENTS, world_dim=WORLD_DIM, obs_dim=OBS_DIM,
        msg_dim=MSG_DIM, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        entropy_start=ENTROPY_WEIGHT_START, entropy_decay=ENTROPY_DECAY_EPISODES,
        comm_cost_end=COMM_COST_END, comm_cost_ramp=COMM_COST_RAMP_EPISODES,
        timesteps=TIMESTEPS_PER_EPISODE,
        topology=TOPOLOGY, k_start=K_NEIGHBORS_START, k_end=K_NEIGH_MIN if 'K_NEIGH_MIN' in globals() else K_NEIGH_MAX if 'K_NEIGH_MAX' in globals() else K_NEIGHBORS_END,
        selection_interval=SELECTION_INTERVAL,
        oscillation_amplitude=OSCILLATION_AMPLITUDE
    )
    
    # Start with maximum k for curriculum topology
    current_k = K_NEIGHBORS_START
    neighbors = get_topology(NUM_AGENTS, TOPOLOGY, current_k)
    
    if verbose:
        print(f"Topology: {TOPOLOGY}, Agents: {NUM_AGENTS}, K: {current_k} (will decay to {K_NEIGHBORS_END})")
        for i, nbs in neighbors.items():
            print(f"  Agent {i} receives from: {nbs}")
    
    agents = []
    for i in range(NUM_AGENTS):
        num_neighbors = len(neighbors[i])
        agent = Agent(OBS_DIM, MSG_DIM, VOCAB_SIZE, HIDDEN_DIM, LATENT_DIM, 
                     num_neighbors, NUM_AGENTS, ACTION_DIM)
        agents.append(agent)
    
    tau = TAU_START
    
    # Track per-agent fitness for survival selection
    agent_fitness = [0.0] * NUM_AGENTS

    exploration_std = EXPLORATION_STD_START
    policy_temp = POLICY_TEMP_START

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes_list = []
    recon_losses_list = []
    pred_losses_list = []
    reward_losses_list = []
    line_recon, = ax.plot([], [], 'b-', label='Avg Recon Loss')
    line_pred, = ax.plot([], [], 'g-', label='Avg Pred Loss')
    line_reward, = ax.plot([], [], 'r-', label='Avg Reward Loss')
    ax.set_title("Training Progress (Real-Time)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    for episode in range(num_episodes):
        # Reset LSTM hidden states at start of each episode (new trajectory)
        for agent in agents:
            agent.reset_hidden()
        
        total_ssl_loss = 0
        total_rl_loss = 0
        recon_losses_episode = []
        pred_losses_episode = []
        reward_losses_episode = []
        trajectory_rewards = []
        
        # Intermediate structures to store data for RL backward pass
        episode_data = [] # List of tuples: (received_msgs_list, obs_list, msgs_hard, msgs_soft, z_tp1, reward)

        # Calculate curriculum weights
        e_progress = min(1.0, episode / ENTROPY_DECAY_EPISODES)
        entropy_weight = ENTROPY_WEIGHT_START - (ENTROPY_WEIGHT_START - ENTROPY_WEIGHT_END) * e_progress
        c_progress = min(1.0, episode / COMM_COST_RAMP_EPISODES)
        comm_cost_weight = COMM_COST_START + (COMM_COST_END - COMM_COST_START) * c_progress
        
        # Initial z_t for the rollout
        z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        
        # 1. ROLLOUT PHASE (Collect trajectory)
        for t in range(TIMESTEPS_PER_EPISODE):
            current_world = z_t @ PROJECTION_MAT
            obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
            
            # Generate actions and messages with exploration
            actions = [agents[i].generate_action(obs_list[i], exploration_std) for i in range(NUM_AGENTS)]
            aggregated_actions = torch.cat(actions, dim=-1)
            
            _, _, _, z_tp1, reward = generate_world_batch(BATCH_SIZE, WORLD_DIM, aggregated_actions, z_t)
            trajectory_rewards.append(reward.mean().item())
            
            msgs_hard_soft = [agents[i].generate_message(obs_list[i], tau, policy_temp) for i in range(NUM_AGENTS)]
            msgs_hard = [hs[0] for hs in msgs_hard_soft]
            msgs_soft = [hs[1] for hs in msgs_hard_soft]
            
            received_msgs_list = []
            for i in range(NUM_AGENTS):
                received = [msgs_hard[j] for j in neighbors[i]]
                received_msgs_list.append(received)
            
            # Store data for backward pass
            episode_data.append((received_msgs_list, obs_list, msgs_hard, msgs_soft, z_tp1, reward))
            z_t = z_tp1.detach()

        # 2. OPTIMIZATION PHASE
        # Compute returns for the trajectory
        returns = compute_returns(trajectory_rewards) # (TIMESTEPS,)

        for t in range(TIMESTEPS_PER_EPISODE):
            received_msgs_list, obs_list, msgs_hard, msgs_soft, z_tp1, reward = episode_data[t]
            ret = returns[t]

            for i in range(NUM_AGENTS):
                ssl, rl, recon, pred, _ = agents[i].compute_loss(
                    received_msgs_list[i], obs_list[i], msgs_hard[i], msgs_soft[i], 
                    z_tp1, ret, episode, entropy_weight, comm_cost_weight
                )
                total_ssl_loss += ssl
                total_rl_loss += rl
                
                recon_losses_episode.append(recon)
                pred_losses_episode.append(pred)
                reward_losses_episode.append(trajectory_rewards[t])

        # Backward: World Model (SSL)
        total_ssl_loss.backward(retain_graph=True)
        for agent in agents:
            # Clip gradients for world_model and decoder (which is part of SSL)
            torch.nn.utils.clip_grad_norm_(list(agent.world_model.parameters()) + list(agent.decoder.parameters()), MAX_GRAD_NORM)
            agent.world_model_optimizer.step()
            agent.world_model_optimizer.zero_grad()
        
        # Backward: Policy (RL)
        total_rl_loss.backward()
        for agent in agents:
            torch.nn.utils.clip_grad_norm_(agent.policy_optimizer.parameters(), MAX_GRAD_NORM)
            agent.policy_optimizer.step()
            agent.policy_optimizer.zero_grad()

        if episode % ANNEAL_EVERY == 0:
            tau = max(0.01, tau * ANNEAL_RATE)
            exploration_std = max(EXPLORATION_STD_MIN, exploration_std * STOCHASTIC_DECAY_RATE)
            policy_temp = max(POLICY_TEMP_MIN, policy_temp * STOCHASTIC_DECAY_RATE)
        
        for agent in agents:
            agent.world_model_scheduler.step()
            agent.policy_scheduler.step()
        
        if episode % 10 == 0:
            episodes_list.append(episode)
            avg_recon = np.mean(recon_losses_episode)
            avg_pred = np.mean(pred_losses_episode)
            avg_reward = np.mean(reward_losses_episode)
            recon_losses_list.append(avg_recon)
            pred_losses_list.append(avg_pred)
            reward_losses_list.append(avg_reward)
            
            # Update per-agent fitness (negative loss = higher fitness)
            for i in range(NUM_AGENTS):
                agent_fitness[i] = -np.mean(recon_losses_episode) - np.mean(pred_losses_episode)
            
            # Log to file
            logger.log_episode(episode, avg_recon, avg_pred, avg_reward, tau)
            
            line_recon.set_data(episodes_list, recon_losses_list)
            line_pred.set_data(episodes_list, pred_losses_list)
            line_reward.set_data(episodes_list, reward_losses_list)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if verbose:
                print(f"Episode {episode}: Avg Recon Loss = {avg_recon:.4f}, Avg Pred Loss = {avg_pred:.4f}, Avg Reward Loss = {avg_reward:.4f}, Tau = {tau:.4f}, K = {current_k}")
        
        # Curriculum topology: update k every 100 episodes
        if episode > 0 and episode % 100 == 0:
            new_k = get_current_k(episode)
            if new_k != current_k:
                current_k = new_k
                neighbors = get_topology(NUM_AGENTS, TOPOLOGY, current_k)
                if verbose:
                    print(f"  [Curriculum] K decreased to {current_k}")
        
        # Survival selection: every SELECTION_INTERVAL episodes
        if episode > 0 and episode % SELECTION_INTERVAL == 0:
            agents, elites, replaced = survival_selection(agents, agent_fitness)
            if verbose:
                print(f"  [Selection] Cloned agents {elites} → replaced {replaced}")
        
        # Periodic evaluation every 200 episodes to monitor emergence live
        if episode > 0 and episode % 200 == 0:
            print(f"\n{'='*50}")
            print(f"PERIODIC EVALUATION (Episode {episode})")
            print(f"{'='*50}")
            eval_result = evaluate(agents, tau, neighbors, verbose=True)
            # Log evaluation
            logger.log_evaluation(episode, {
                "vocab_used": eval_result.get("vocab_used", 0),
                "top_symbols": eval_result.get("top_symbols", []),
                "hub_counts": {},
                "recon_loss": eval_result.get("recon_loss", 0),
                "pred_loss": eval_result.get("pred_loss", 0),
                "current_k": current_k
            })

    plt.ioff()
    plt.savefig('training_progress.png')
    print("Training progress plot saved to training_progress.png")

    # Final metrics and save log
    logger.log_final({
        "final_recon_loss": recon_losses_list[-1] if recon_losses_list else 0,
        "final_pred_loss": pred_losses_list[-1] if pred_losses_list else 0,
        "final_tau": tau,
        "total_episodes": num_episodes
    })
    logger.save()

    if verbose:
        print("Training complete.")
    return agents, tau, neighbors, logger


def evaluate(agents, tau, neighbors, verbose=True):
    with torch.no_grad():
        # CRITICAL: Use concatenated actions (all zeros for eval is fine but must be correct size)
        aggregated_actions = torch.zeros(BATCH_SIZE, NUM_AGENTS * ACTION_DIM, device=device)
        current_world, target_world, z_t, z_tp1, reward = generate_world_batch(BATCH_SIZE, WORLD_DIM, aggregated_actions)
        obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
        
        # evaluation uses policy_temp = 1.0 (or min)
        msgs_hard_soft = [agents[i].generate_message(obs_list[i], tau, temperature=1.0) for i in range(NUM_AGENTS)]
        msgs_hard = [hs[0] for hs in msgs_hard_soft]
        msgs_soft = [hs[1] for hs in msgs_hard_soft]
        symbols_list = [msg.argmax(dim=-1).cpu().numpy() for msg in msgs_hard]
        
        if verbose:
            print("\n--- Model Inspection ---")
            print("Sample Symbolic Messages (first 3 batches, first 3 agents):")
            for agent_id in range(min(3, NUM_AGENTS)):
                print(f"Agent {agent_id}: {symbols_list[agent_id][:3]}")

        received_msgs_list = []
        for i in range(NUM_AGENTS):
            received = [msgs_hard[j] for j in neighbors[i]]
            received_msgs_list.append(received)
        
        recon_losses = []
        pred_losses = []
        for i in range(NUM_AGENTS):
            # Pass dummy weights and dummy scalar return for evaluation
            # returns_trajectory is expected as a scalar or matching batch in compute_loss
            # but compute_loss does: rl_loss = -torch.mean(msg_logprobs * returns_trajectory)
            # so returns_trajectory can be a scalar if we want.
            _, _, r_loss, p_loss, _ = agents[i].compute_loss(
                received_msgs_list[i], obs_list[i], msgs_hard[i], msgs_soft[i], 
                z_tp1, torch.zeros(1, device=device), 0, 0.0, 0.0
            )
            recon_losses.append(r_loss)
            pred_losses.append(p_loss)
        
        if verbose:
            print(f"Avg Reconstruction Loss: {np.mean(recon_losses):.4f}")
            print(f"Avg Prediction Loss: {np.mean(pred_losses):.4f}")
        
        all_symbols = np.concatenate([sym.ravel() for sym in symbols_list])
        freq = collections.Counter(all_symbols)
        if verbose:
            print(f"Used {len(freq)}/{VOCAB_SIZE} symbols")
            print("Top 5 symbols:", freq.most_common(5))
        
        one_hot_symbols = np.zeros((BATCH_SIZE * NUM_AGENTS * MSG_DIM, VOCAB_SIZE))
        flat_symbols = all_symbols.ravel()
        one_hot_symbols[np.arange(len(flat_symbols)), flat_symbols] = 1
        flat_z = z_t.cpu().numpy().repeat(NUM_AGENTS * MSG_DIM, axis=0)
        corrs = np.corrcoef(one_hot_symbols.T, flat_z.T)[:VOCAB_SIZE, VOCAB_SIZE:]
        strong_corrs = np.abs(corrs) > 0.5
        
        if verbose:
            if np.any(strong_corrs):
                print("Strong Symbol-Latent Correlations (>0.5 abs):")
                for sym in range(VOCAB_SIZE):
                    for dim in range(LATENT_DIM):
                        if strong_corrs[sym, dim]:
                            print(f"  Symbol {sym} <-> Latent Dim {dim}: {corrs[sym, dim]:.4f}")
            else:
                print("No strong symbol-latent correlations (>0.5 abs) detected.")
        
        # ============ EMERGENCE DIAGNOSTICS ============
        
        # 1. Attention Patterns - Hub/Relay Detection
        if verbose:
            print("\n--- Attention Patterns (Hub/Relay Detection) ---")
            hub_counts = collections.Counter()
            for i in range(NUM_AGENTS):
                if agents[i].last_attn_weights is not None:
                    avg_weights = agents[i].last_attn_weights.mean(dim=0).cpu().numpy()
                    hub_idx = np.argmax(avg_weights)
                    if avg_weights[hub_idx] > 0.2:  # Threshold for 'hub'
                        actual_neighbor = neighbors[i][hub_idx]
                        hub_counts[actual_neighbor] += 1
                        print(f"Agent {i}: Attends to neighbor {actual_neighbor} with weight {avg_weights[hub_idx]:.4f}")
            
            if hub_counts:
                top_hubs = hub_counts.most_common(3)
                print(f"Potential Hub Agents (most attended): {top_hubs}")
            else:
                print("No clear hub/relay patterns detected (attention distributed).")
        
        # 2. Symbol Specialization - Vocab Narrowing
        if verbose:
            print("\n--- Symbol Specialization ---")
            current_vocab_used = len(freq)
            total_symbols = BATCH_SIZE * NUM_AGENTS * MSG_DIM
            dominance = sum(count for _, count in freq.most_common(10)) / total_symbols
            
            if current_vocab_used < VOCAB_SIZE * 0.7 or dominance > 0.5:
                print(f"Specialization detected: {current_vocab_used}/{VOCAB_SIZE} symbols, top 10 dominate {dominance:.1%}")
                specialized = [sym for sym, _ in freq.most_common(10)]
                print(f"  Top 10 symbols: {specialized}")
            else:
                print(f"Vocab diverse: {current_vocab_used}/{VOCAB_SIZE} symbols, top 10 = {dominance:.1%}")
        
        # 3. Temporal Patterns - LSTM Grammar
        if verbose:
            print("\n--- Temporal Patterns (LSTM Grammar) ---")
            # Reset hidden states for fresh rollout
            for agent in agents:
                agent.reset_hidden()
            
            z_t_temp = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            symbol_sequences = [[] for _ in range(NUM_AGENTS)]
            
            for t in range(5):
                current_world_temp = z_t_temp @ PROJECTION_MAT
                obs_list_temp = [get_partial_obs(current_world_temp, i, OBS_DIM) for i in range(NUM_AGENTS)]
                
                for i in range(NUM_AGENTS):
                    msg_hard, _ = agents[i].generate_message(obs_list_temp[i], tau)
                    symbol = msg_hard[0, 0].argmax().item()  # batch 0, msg_dim 0
                    symbol_sequences[i].append(symbol)
                
                # Simple evolve
                z_t_temp = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            
            grammar_detected = False
            for i in range(min(3, NUM_AGENTS)):  # Check first 3 agents
                seq = symbol_sequences[i]
                print(f"Agent {i} sequence: {seq}")
                
                # Check for repetitions
                unique_ratio = len(set(seq)) / len(seq)
                if unique_ratio < 0.6:
                    print(f"  Pattern: Repetitions detected (unique ratio: {unique_ratio:.2f})")
                    grammar_detected = True
                
                # Check sequential correlation
                if len(seq) > 1:
                    try:
                        corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                        if abs(corr) > 0.3:
                            print(f"  Pattern: Sequential correlation {corr:.2f}")
                            grammar_detected = True
                    except:
                        pass
            
            if not grammar_detected:
                print("No clear temporal patterns detected yet.")
        
        return {
            'recon_loss': np.mean(recon_losses),
            'pred_loss': np.mean(pred_losses),
            'vocab_used': len(freq),
            'top_symbols': freq.most_common(5),
            'strong_correlations': strong_corrs.sum()
        }


def save_agents(agents, prefix="agent"):
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"{prefix}_{i}.pth")
    print(f"Models saved as {prefix}_*.pth")


if __name__ == "__main__":
    import argparse
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
    
    trained_agents, final_tau, neighbors, logger = train()
    visualize_topology(neighbors, NUM_AGENTS)
    evaluate(trained_agents, final_tau, neighbors)
    save_agents(trained_agents)
    
    # LLM Interpretation
    try:
        from llm_bridge import run_llm_interpretation
        print(f"\nAttempting LLM interpretation (provider: {llm_provider})...")
        llm_result = run_llm_interpretation(trained_agents, neighbors, final_tau, device, provider=llm_provider)
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