import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# RTX 4090 Optimizations
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmuls
    torch.backends.cudnn.allow_tf32 = True
    print("CUDA optimizations enabled: cudnn.benchmark, TF32")

# Hyperparameters
NUM_AGENTS = 10  # Scaled up for sparse communication
WORLD_DIM = 10
OBS_DIM = 4
MSG_DIM = 5
HIDDEN_DIM = 64  # Increased for more capacity
NUM_EPISODES = 2000
LEARNING_RATE = 0.001  # Reduced from 0.01 to prevent gradient explosion
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold
BATCH_SIZE = 256  # Increased for RTX 4090 (24GB VRAM)
VOCAB_SIZE = 64
TAU_START = 1.0
ANNEAL_RATE = 0.995
ANNEAL_EVERY = 10
ENTROPY_WEIGHT = 0.01
ACTION_DIM = 4  # New: Dim for agent actions (e.g., forces on particles)
REWARD_WEIGHT = 1.0  # Weight for cooperative reward loss
COMM_COST_WEIGHT = 0.001  # L1 penalty on message to encourage sparsity
TIMESTEPS_PER_EPISODE = 5  # New for multi-step rollouts (priority 1)

# Topology settings
TOPOLOGY = 'knearest'
K_NEIGHBORS = 4

LATENT_DIM = 8
NUM_PARTICLES = LATENT_DIM // 2

PROJECTION_MAT = torch.randn(LATENT_DIM, WORLD_DIM).to(device)

COHESION_FACTOR = 0.01
SEPARATION_FACTOR = 0.05
NOISE_FACTOR = 0.05
ACTION_SCALE = 0.1  # Scale factor for applying actions to latent


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
        
        # Project messages to query/key/value
        self.query_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.key_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.value_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        
        # LayerNorm for stability (prevents variance explosion)
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_key = nn.LayerNorm(hidden_dim)
        self.norm_value = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, own_msg, received_msgs):
        batch_size = own_msg.size(0)
        num_neighbors = len(received_msgs)
        
        # Flatten messages
        own_flat = own_msg.view(batch_size, -1)  # (B, msg_flat_dim)
        
        # Stack neighbors: (B, num_neighbors, msg_flat_dim)
        neighbors_flat = torch.stack([m.view(batch_size, -1) for m in received_msgs], dim=1)
        
        # Query from own message, keys/values from neighbors (with LayerNorm)
        query = self.norm_query(self.query_proj(own_flat)).unsqueeze(1)  # (B, 1, hidden_dim)
        keys = self.norm_key(self.key_proj(neighbors_flat))  # (B, num_neighbors, hidden_dim)
        values = self.norm_value(self.value_proj(neighbors_flat))  # (B, num_neighbors, hidden_dim)
        
        # Multi-head attention
        attn_out, attn_weights = self.attention(query, keys, values)
        attn_out = attn_out.squeeze(1)  # (B, hidden_dim)
        
        # Output projection with LayerNorm
        out = self.norm_out(self.out_proj(attn_out))
        
        return out, attn_weights.squeeze(1)  # (B, hidden_dim), (B, num_neighbors)


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


class Agent(nn.Module):
    def __init__(self, obs_dim, msg_dim, vocab_size, hidden_dim, latent_dim, 
                 num_neighbors, total_agents, action_dim, use_lstm=True):
        super(Agent, self).__init__()
        self.use_lstm = use_lstm
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        self.encoder = LSTMEncoder(obs_dim, hidden_dim, msg_dim, vocab_size)
        self.hidden_state = None  # Will store (h, c) for LSTM
        
        self.action_head = ActionHead(obs_dim, hidden_dim, action_dim)
        # Use AttentionDecoder instead of Decoder
        self.decoder = AttentionDecoder(msg_dim, vocab_size, hidden_dim, obs_dim, num_heads=4)
        self.last_attn_weights = None  # Store for analysis
        
        # World estimator also uses attention
        self.world_attention = MessageAttention(msg_dim, vocab_size, hidden_dim, num_heads=4)
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
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
        # Warm-up LR scheduler (100 episode linear ramp, then constant)
        def warm_up_lambda(epoch):
            if epoch < 100:
                return epoch / 100.0  # Linear warm-up
            return 1.0
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_lambda)
        
        self.to(device)  # Move model to device
        
        # Compile model for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            self.world_attention = torch.compile(self.world_attention)
    
    def reset_hidden(self):
        """Reset hidden state at the start of a new trajectory."""
        self.hidden_state = None
    
    def generate_message(self, obs, tau):
        logits, self.hidden_state = self.encoder(obs, self.hidden_state)
        hard, soft = gumbel_softmax(logits, tau, hard=True)
        return hard, soft
    
    def generate_action(self, obs):
        return self.action_head(obs)
    
    def reconstruct(self, own_msg, received_msgs):
        """Now uses attention over received messages, with own_msg as query."""
        recon, attn_weights = self.decoder(own_msg, received_msgs)
        self.last_attn_weights = attn_weights
        return recon
    
    def estimate_latent(self, own_msg, received_msgs):
        """Use attention to aggregate messages for world estimation."""
        attended, _ = self.world_attention(own_msg, received_msgs)
        return self.world_proj(attended)
    
    def predict_next(self, hat_z):
        return self.transition(hat_z)
    
    def compute_loss(self, received_msgs, true_obs, own_msg_hard, own_msg_soft, target_z, reward):
        # Reconstruction now uses attention (own_msg as query)
        recon = self.reconstruct(own_msg_hard, received_msgs)
        recon_loss = nn.MSELoss()(recon, true_obs)
        
        # World estimation uses attention too
        hat_z = self.estimate_latent(own_msg_hard, received_msgs)
        hat_z_next = self.predict_next(hat_z)
        pred_loss = nn.MSELoss()(hat_z_next, target_z)
        
        # Entropy bonus (encourage exploration of vocabulary)
        entropy = -torch.mean(own_msg_soft * torch.log(own_msg_soft + 1e-10))
        entropy_loss = -ENTROPY_WEIGHT * entropy
        
        # Communication cost: L1 penalty on message (encourage sparsity/conciseness)
        comm_cost = COMM_COST_WEIGHT * torch.mean(torch.abs(own_msg_soft))
        
        reward_loss = -REWARD_WEIGHT * reward.mean()  # Maximize reward (negative loss)
        
        total_loss = recon_loss + pred_loss + entropy_loss + comm_cost + reward_loss
        
        # NaN check for debugging
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"WARNING: NaN/Inf detected! recon={recon_loss.item():.4f}, pred={pred_loss.item():.4f}")
            total_loss = torch.zeros_like(total_loss)  # Skip this batch
        
        return total_loss, recon_loss, pred_loss, reward_loss.item(), comm_cost.item()
    
    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'action_head': self.action_head.state_dict(),
            'decoder': self.decoder.state_dict(),
            'world_attention': self.world_attention.state_dict(),
            'world_proj': self.world_proj.state_dict(),
            'transition': self.transition.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.action_head.load_state_dict(state_dict['action_head'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.world_attention.load_state_dict(state_dict['world_attention'])
        self.world_proj.load_state_dict(state_dict['world_proj'])
        self.transition.load_state_dict(state_dict['transition'])


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
    
    action_delta = aggregated_actions.view(-1, 2, 2) * ACTION_SCALE  # (B, 2 particles, 2 dims)
    action_applied = torch.zeros_like(positions)
    action_applied[:, :2] += action_delta  # Apply to first 2 particles
    
    delta = cohesion + separation + NOISE_FACTOR * torch.randn_like(positions) + action_applied
    new_positions = positions + delta
    
    # Compute reward: negative variance (encourage clustering)
    reward = -new_positions.var(dim=1).mean(dim=-1)
    
    return new_positions.view(-1, LATENT_DIM), reward


def generate_world_batch(batch_size, world_dim, aggregated_actions, z_t=None):
    if z_t is None:
        z_t = torch.randn(batch_size, LATENT_DIM, device=device)
    current_world = z_t @ PROJECTION_MAT
    z_tp1, reward = apply_swarm_dynamics(z_t, aggregated_actions)
    target_world = z_tp1 @ PROJECTION_MAT
    return current_world, target_world, z_t, z_tp1, reward


def get_partial_obs(world, agent_id, obs_dim):
    start = (agent_id * obs_dim) % world.shape[1]
    indices = torch.arange(start, start + obs_dim, device=device) % world.shape[1]
    return world[:, indices]


# ============ TRAINING ============

def train(num_episodes=NUM_EPISODES, verbose=True):
    neighbors = get_topology(NUM_AGENTS, TOPOLOGY, K_NEIGHBORS)
    
    if verbose:
        print(f"Topology: {TOPOLOGY}, Agents: {NUM_AGENTS}")
        for i, nbs in neighbors.items():
            print(f"  Agent {i} receives from: {nbs}")
    
    agents = []
    for i in range(NUM_AGENTS):
        num_neighbors = len(neighbors[i])
        agent = Agent(OBS_DIM, MSG_DIM, VOCAB_SIZE, HIDDEN_DIM, LATENT_DIM, 
                     num_neighbors, NUM_AGENTS, ACTION_DIM)
        agents.append(agent)
    
    tau = TAU_START

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes_list = []
    recon_losses_list = []
    pred_losses_list = []
    reward_losses_list = []  # New: Track reward loss
    line_recon, = ax.plot([], [], 'b-', label='Avg Recon Loss')
    line_pred, = ax.plot([], [], 'g-', label='Avg Pred Loss')
    line_reward, = ax.plot([], [], 'r-', label='Avg Reward Loss')
    ax.set_title("Training Progress (Real-Time)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    
    # Mixed precision for RTX 4090
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for episode in range(num_episodes):
        # Reset LSTM hidden states at start of each episode (new trajectory)
        for agent in agents:
            agent.reset_hidden()
        
        total_loss = 0
        recon_losses_episode = []
        pred_losses_episode = []
        reward_losses_episode = []
        
        # Initial z_t for the rollout
        z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        
        for t in range(TIMESTEPS_PER_EPISODE):
            # Get current obs
            current_world = z_t @ PROJECTION_MAT
            obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
            
            # Generate actions
            actions = [agents[i].generate_action(obs_list[i]) for i in range(NUM_AGENTS)]
            aggregated_actions = torch.mean(torch.stack(actions), dim=0)  # Collective action
            
            # Evolve world (using current z_t and actions to get next)
            _, _, _, z_tp1, reward = generate_world_batch(BATCH_SIZE, WORLD_DIM, aggregated_actions, z_t)
            
            # Generate messages based on current obs
            msgs_hard_soft = [agents[i].generate_message(obs_list[i], tau) for i in range(NUM_AGENTS)]
            msgs_hard = [hs[0] for hs in msgs_hard_soft]
            msgs_soft = [hs[1] for hs in msgs_hard_soft]
            
            # Build received messages
            received_msgs_list = []
            for i in range(NUM_AGENTS):
                received = [msgs_hard[j] for j in neighbors[i]]
                received_msgs_list.append(received)
            
            # Compute losses for this timestep
            recon_losses_timestep = []
            pred_losses_timestep = []
            reward_losses_timestep = []
            
            timestep_loss = 0
            for i in range(NUM_AGENTS):
                loss, recon_loss, pred_loss, r_loss, c_cost = agents[i].compute_loss(received_msgs_list[i], obs_list[i], msgs_hard[i], msgs_soft[i], z_tp1, reward)
                timestep_loss += loss
                recon_losses_timestep.append(recon_loss.item())
                pred_losses_timestep.append(pred_loss.item())
                reward_losses_timestep.append(r_loss)
            
            total_loss += timestep_loss
            
            recon_losses_episode.extend(recon_losses_timestep)
            pred_losses_episode.extend(pred_losses_timestep)
            reward_losses_episode.extend(reward_losses_timestep)
            
            # Update z_t for next timestep (detach to break gradient chain)
            z_t = z_tp1.detach()
        
        # Backward and step after full rollout (with AMP + gradient clipping)
        scaler.scale(total_loss).backward()
        for agent in agents:
            scaler.unscale_(agent.optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            scaler.step(agent.optimizer)
        scaler.update()
        
        if episode % ANNEAL_EVERY == 0:
            tau = max(0.01, tau * ANNEAL_RATE)
        
        # Scheduler step every episode (for warm-up to work properly)
        for agent in agents:
            agent.scheduler.step()
        
        if episode % 10 == 0:
            episodes_list.append(episode)
            avg_recon = np.mean(recon_losses_episode)
            avg_pred = np.mean(pred_losses_episode)
            avg_reward = np.mean(reward_losses_episode)
            recon_losses_list.append(avg_recon)
            pred_losses_list.append(avg_pred)
            reward_losses_list.append(avg_reward)
            
            line_recon.set_data(episodes_list, recon_losses_list)
            line_pred.set_data(episodes_list, pred_losses_list)
            line_reward.set_data(episodes_list, reward_losses_list)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if verbose:
                print(f"Episode {episode}: Avg Recon Loss = {avg_recon:.4f}, Avg Pred Loss = {avg_pred:.4f}, Avg Reward Loss = {avg_reward:.4f}, Tau = {tau:.4f}")

    plt.ioff()
    plt.savefig('training_progress.png')
    print("Training progress plot saved to training_progress.png")

    if verbose:
        print("Training complete.")
    return agents, tau, neighbors


def evaluate(agents, tau, neighbors, verbose=True):
    with torch.no_grad():
        aggregated_actions = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)  # Dummy for eval
        current_world, target_world, z_t, z_tp1, reward = generate_world_batch(BATCH_SIZE, WORLD_DIM, aggregated_actions)
        obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
        msgs_hard_soft = [agents[i].generate_message(obs_list[i], tau) for i in range(NUM_AGENTS)]
        msgs_hard = [hs[0] for hs in msgs_hard_soft]
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
            _, recon_loss, pred_loss, _ = agents[i].compute_loss(received_msgs_list[i], obs_list[i], msgs_hard[i], torch.zeros_like(msgs_hard[i]), z_tp1, reward)
            recon_losses.append(recon_loss.item())
            pred_losses.append(pred_loss.item())
        
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
    trained_agents, final_tau, neighbors = train()
    visualize_topology(neighbors, NUM_AGENTS)
    evaluate(trained_agents, final_tau, neighbors)
    save_agents(trained_agents)
    print("Done.")