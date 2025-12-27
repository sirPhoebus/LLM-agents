import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils


class LSTMEncoder(nn.Module):
    """Vectorized LSTM-based Encoder for multiple agents."""
    def __init__(self, input_dim, hidden_dim, msg_dim, vocab_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        
        # Project observation to LSTM input space
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM for temporal memory
        # We process (num_agents * batch_size, 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output projection to message logits
        self.output_proj = nn.Linear(hidden_dim, msg_dim * vocab_size)
    
    def init_hidden(self, num_agents, batch_size, device):
        """Initialize hidden state for all agents."""
        h0 = torch.zeros(self.num_layers, num_agents * batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, num_agents * batch_size, self.hidden_dim, device=device)
        return (h0, c0)
    
    def forward(self, x, hidden=None):
        """
        x: (num_agents, batch_size, input_dim)
        hidden: (h, c) each (num_layers, num_agents * batch_size, hidden_dim)
        """
        num_agents, batch_size, input_dim = x.shape
        device = x.device
        
        # Flatten num_agents and batch_size
        x_flat = x.view(num_agents * batch_size, input_dim)
        
        if hidden is None:
            hidden = self.init_hidden(num_agents, batch_size, device)
        
        # Project and add sequence dimension
        x_proj = torch.relu(self.input_proj(x_flat))  # (N*B, hidden_dim)
        x_seq = x_proj.unsqueeze(1)  # (N*B, 1, hidden_dim)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x_seq, hidden)  # (N*B, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (N*B, hidden_dim)
        
        # Project to message logits
        logits = self.output_proj(lstm_out).view(num_agents, batch_size, self.msg_dim, self.vocab_size)
        
        return logits, hidden


class ActionHead(nn.Module):
    """Vectorized Action Head."""
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """x: (num_agents, batch_size, obs_dim)"""
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class SwarmAttention(nn.Module):
    """Vectorized Attention over all agents with topology masking."""
    def __init__(self, msg_dim, vocab_size, hidden_dim, num_heads=4):
        super(SwarmAttention, self).__init__()
        self.msg_flat_dim = msg_dim * vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.learned_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.query_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.key_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        self.value_proj = nn.Linear(self.msg_flat_dim, hidden_dim)
        
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_key = nn.LayerNorm(hidden_dim)
        self.norm_value = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query_msgs, key_msgs, mask=None):
        """
        query_msgs: (num_agents, batch_size, msg_flat_dim) or None
        key_msgs: (num_agents, batch_size, msg_flat_dim)
        mask: (num_agents, num_agents) - topology mask (True/1 = allowed)
        """
        num_agents, batch_size, _ = key_msgs.shape
        
        # Reshape to (batch_size, num_agents, dim)
        keys_flat = key_msgs.transpose(0, 1)  # (B, N, D)
        
        if query_msgs is None:
            # Shared learned query for all agents
            query = self.norm_query(self.learned_query).expand(batch_size, num_agents, -1)
        else:
            query_flat = query_msgs.transpose(0, 1)
            query = self.norm_query(self.query_proj(query_flat))
        
        keys = self.norm_key(self.key_proj(keys_flat))
        values = self.norm_value(self.value_proj(keys_flat))
        
        # Prepare mask for MultiheadAttention (needs to be (B*heads, N_query, N_key))
        # Or (N_query, N_key) which will be broadcast.
        # MHA mask uses True/1 to MASK OUT. Our mask is 1 = allow.
        # So conversion: attn_mask = (1 - mask).bool()
        attn_mask = None
        if mask is not None:
            attn_mask = (1.0 - mask).bool()
        
        attn_out, attn_weights = self.attention(query, keys, values, attn_mask=attn_mask)
        
        # Output: (B, N, D) -> (N, B, D)
        out = self.norm_out(self.out_proj(attn_out)).transpose(0, 1)
        
        return out, attn_weights


class SwarmWorldModel(nn.Module):
    """Vectorized World Model for the swarm."""
    def __init__(self, msg_dim, vocab_size, hidden_dim, latent_dim, num_heads=4):
        super(SwarmWorldModel, self).__init__()
        self.attention = SwarmAttention(msg_dim, vocab_size, hidden_dim, num_heads)
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

    def forward(self, all_msgs, mask):
        """
        all_msgs: (num_agents, batch_size, msg_dim, vocab_size)
        mask: (num_agents, num_agents)
        """
        n, b, m, v = all_msgs.shape
        msgs_flat = all_msgs.view(n, b, -1)
        
        # Use mean of all messages as a base for query if no own_msg is passed 
        # (or just use learned query from attention module)
        attended, _ = self.attention(None, msgs_flat, mask=mask)
        hat_z = self.world_proj(attended)
        hat_z_next = self.transition(hat_z)
        return hat_z_next, hat_z


class SwarmDecoder(nn.Module):
    """Vectorized Decoder for all agents."""
    def __init__(self, msg_dim, vocab_size, hidden_dim, output_dim, num_heads=4):
        super(SwarmDecoder, self).__init__()
        self.attention = SwarmAttention(msg_dim, vocab_size, hidden_dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, all_msgs, mask):
        n, b, m, v = all_msgs.shape
        msgs_flat = all_msgs.view(n, b, -1)
        
        attended, attn_weights = self.attention(None, msgs_flat, mask=mask)
        recon = self.fc(attended)
        return recon, attn_weights


class Swarm(nn.Module):
    """
    Centralized model managing all agents in a vectorized manner.
    Optimizes speed by batching across agents.
    """
    def __init__(self, obs_dim, msg_dim, vocab_size, hidden_dim, latent_dim, 
                 total_agents, action_dim, batch_size, learning_rate):
        super(Swarm, self).__init__()
        self.total_agents = total_agents
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        # Shared or individualized components? 
        # Here we use shared weights for the policy (homogenous agents) 
        # but individualized hidden states. 
        # This is common in MAPPO/emergent comm.
        self.policy_encoder = LSTMEncoder(obs_dim, hidden_dim, msg_dim, vocab_size)
        self.action_head = ActionHead(obs_dim, hidden_dim, action_dim)
        
        self.world_model = SwarmWorldModel(msg_dim, vocab_size, hidden_dim, latent_dim)
        self.decoder = SwarmDecoder(msg_dim, vocab_size, hidden_dim, obs_dim)
        
        self.hidden_state = None
        self.last_attn_weights = None

        # Group optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / 100.0)
        )

    def reset_hidden(self, device):
        self.hidden_state = self.policy_encoder.init_hidden(self.total_agents, self.batch_size, device)

    def forward_policy(self, obs, tau, temperature=1.0, exploration_std=0.1):
        """
        obs: (total_agents, batch_size, obs_dim)
        Returns: msgs_hard, msgs_soft, actions
        """
        msg_logits, self.hidden_state = self.policy_encoder(obs, self.hidden_state)
        action_mean = self.action_head(obs)
        
        # Messages
        scaled_logits = msg_logits / temperature
        msgs_hard, msgs_soft = utils.gumbel_softmax(scaled_logits, tau, hard=True)
        
        # Actions
        noise = torch.randn_like(action_mean) * exploration_std
        actions = action_mean + noise
        
        return msgs_hard, msgs_soft, actions

    def compute_loss(self, obs, msgs_hard, msgs_soft, returns, target_z, mask, 
                     entropy_weight, comm_cost_weight):
        """
        obs: (N, B, obs_dim)
        msgs_hard/soft: (N, B, msg_dim, vocab_size)
        returns: (N, B) - or scaled returns
        target_z: (B, latent_dim) -> will be expanded to (N, B, latent_dim)
        mask: (N, N)
        """
        device = obs.device
        
        # 1. SSL Losses
        recon, self.last_attn_weights = self.decoder(msgs_hard, mask)
        recon_loss = nn.MSELoss()(recon, obs)
        
        hat_z_next, _ = self.world_model(msgs_hard, mask)
        # target_z is global world state, agents try to predict it from local msgs
        target_z_exp = target_z.unsqueeze(0).expand(self.total_agents, -1, -1)
        pred_loss = nn.MSELoss()(hat_z_next, target_z_exp)
        
        # Penalty/Regularization
        entropy = -torch.mean(msgs_soft * torch.log(msgs_soft + 1e-10))
        entropy_loss = -entropy_weight * entropy
        comm_cost = comm_cost_weight * torch.mean(torch.abs(msgs_soft))
        
        ssl_loss = recon_loss + pred_loss + entropy_loss + comm_cost
        
        # 2. RL Loss (REINFORCE)
        # Log-probs of chosen symbols
        log_probs = -nn.CrossEntropyLoss(reduction='none')(
            msgs_soft.view(-1, self.vocab_size),
            msgs_hard.argmax(-1).view(-1)
        ).view(self.total_agents, self.batch_size, -1).mean(dim=-1)
        
        # returns should be (N, B) or (B,) broadcasted
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)
            
        rl_loss = -torch.mean(log_probs * returns)
        
        total_loss = ssl_loss + rl_loss
        
        return total_loss, recon_loss.item(), pred_loss.item()
