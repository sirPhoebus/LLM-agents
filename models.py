import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import utils


class SpeciesLinear(nn.Module):
    """Linear layer that uses different weights for different species."""
    def __init__(self, num_species, in_features, out_features):
        super().__init__()
        self.num_species = num_species
        self.weight = nn.Parameter(torch.randn(num_species, out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(num_species, out_features))
        
    def forward(self, x, species_indices):
        # x: (num_agents, batch_size, in_features)
        # species_indices: (num_agents,)
        # weights: (num_agents, out_features, in_features)
        w = self.weight[species_indices]
        b = self.bias[species_indices]
        # (N, B, I) @ (N, I, O) -> (N, B, O)
        return torch.matmul(x, w.transpose(-1, -2)) + b.unsqueeze(1)


class LSTMEncoder(nn.Module):
    """Vectorized LSTM-based Encoder for multiple agents, supporting species."""
    def __init__(self, num_species, input_dim, hidden_dim, msg_dim, vocab_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.num_species = num_species
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        
        # Project observation to LSTM input space (species-specific)
        self.input_proj = SpeciesLinear(num_species, input_dim, hidden_dim)
        
        # LSTM for temporal memory (Shared or could be species-specific, keeping shared for speed)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output projection to message logits (species-specific)
        self.output_proj = SpeciesLinear(num_species, hidden_dim, msg_dim * vocab_size)
    
    def init_hidden(self, num_agents, batch_size, device):
        """Initialize hidden state for all agents."""
        h0 = torch.zeros(self.num_layers, num_agents * batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, num_agents * batch_size, self.hidden_dim, device=device)
        return (h0, c0)
    
    def forward(self, x, species_indices, hidden=None):
        """
        x: (num_agents, batch_size, input_dim)
        species_indices: (num_agents,)
        hidden: (h, c) each (num_layers, num_agents * batch_size, hidden_dim)
        """
        num_agents, batch_size, input_dim = x.shape
        device = x.device
        
        if hidden is None:
            hidden = self.init_hidden(num_agents, batch_size, device)
        
        # Project using species identity
        x_proj = torch.relu(self.input_proj(x, species_indices))  # (N, B, hidden_dim)
        
        # Flatten for LSTM (N*B, 1, hidden_dim)
        x_seq = x_proj.view(num_agents * batch_size, 1, self.hidden_dim)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x_seq, hidden)  # (N*B, 1, hidden_dim)
        
        # Reshape for output projection
        lstm_out = lstm_out.view(num_agents, batch_size, self.hidden_dim)
        
        # Project to message logits using species identity
        logits = self.output_proj(lstm_out, species_indices)
        logits = logits.view(num_agents, batch_size, self.msg_dim, self.vocab_size)
        
        return logits, hidden


class ActionHead(nn.Module):
    """Vectorized Action Head, supporting species."""
    def __init__(self, num_species, input_dim, hidden_dim, action_dim):
        super(ActionHead, self).__init__()
        self.fc1 = SpeciesLinear(num_species, input_dim, hidden_dim)
        self.fc2 = SpeciesLinear(num_species, hidden_dim, action_dim)
    
    def forward(self, x, species_indices):
        """x: (num_agents, batch_size, obs_dim)"""
        x = torch.relu(self.fc1(x, species_indices))
        return torch.tanh(self.fc2(x, species_indices))


class SwarmAttention(nn.Module):
    """Vectorized Attention over all agents with topology masking, supporting species."""
    def __init__(self, num_species, msg_dim, vocab_size, hidden_dim, num_heads=4):
        super(SwarmAttention, self).__init__()
        self.msg_flat_dim = msg_dim * vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_species = num_species
        
        # Project messages to QKV space (species-specific)
        self.query_proj = SpeciesLinear(num_species, self.msg_flat_dim, hidden_dim)
        self.key_proj = SpeciesLinear(num_species, self.msg_flat_dim, hidden_dim)
        self.value_proj = SpeciesLinear(num_species, self.msg_flat_dim, hidden_dim)
        
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_key = nn.LayerNorm(hidden_dim)
        self.norm_value = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out_proj = SpeciesLinear(num_species, hidden_dim, hidden_dim)
    
    def forward(self, key_msgs, species_indices, query_msgs=None, mask=None):
        """
        key_msgs: (num_agents, batch_size, msg_flat_dim)
        species_indices: (num_agents,)
        query_msgs: (num_agents, batch_size, msg_flat_dim) or None
        mask: (num_agents, num_agents)
        """
        num_agents, batch_size, _ = key_msgs.shape
        
        # If no query msgs, use zeroes (unconditionally listen)
        if query_msgs is None:
            query_msgs = torch.zeros_like(key_msgs)
            
        # Species-specific projections
        q = self.norm_query(self.query_proj(query_msgs, species_indices))
        k = self.norm_key(self.key_proj(key_msgs, species_indices))
        v = self.norm_value(self.value_proj(key_msgs, species_indices))
        
        # Multihead attention expects (B, N, D)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        attn_mask = None
        if mask is not None:
            attn_mask = (1.0 - mask).bool()
        
        attn_out, attn_weights = self.attention(q, k, v, attn_mask=attn_mask)
        
        # (B, N, D) -> (N, B, D)
        attn_out = attn_out.transpose(0, 1)
        
        # Final projection (species-specific)
        out = self.norm_out(self.out_proj(attn_out, species_indices))
        
        return out, attn_weights


class SwarmWorldModel(nn.Module):
    """Vectorized World Model for the swarm, supporting species."""
    def __init__(self, num_species, msg_dim, vocab_size, hidden_dim, latent_dim, num_heads=4):
        super(SwarmWorldModel, self).__init__()
        self.attention = SwarmAttention(num_species, msg_dim, vocab_size, hidden_dim, num_heads)
        self.world_proj = nn.Sequential(
            SpeciesLinear(num_species, hidden_dim, hidden_dim),
            nn.ReLU(),
            SpeciesLinear(num_species, hidden_dim, latent_dim)
        )
        self.transition = nn.Sequential(
            SpeciesLinear(num_species, latent_dim, hidden_dim),
            nn.ReLU(),
            SpeciesLinear(num_species, hidden_dim, latent_dim)
        )

    def forward(self, all_msgs, species_indices, mask):
        """
        all_msgs: (num_agents, batch_size, msg_dim, vocab_size)
        species_indices: (num_agents,)
        mask: (num_agents, num_agents)
        """
        n, b, m, v = all_msgs.shape
        msgs_flat = all_msgs.view(n, b, -1)
        
        attended, _ = self.attention(msgs_flat, species_indices, mask=mask)
        
        # Apply sequential species blocks manually since nn.Sequential doesn't handle extra args
        x = attended
        for layer in self.world_proj:
            if isinstance(layer, SpeciesLinear):
                x = layer(x, species_indices)
            else:
                x = layer(x)
        hat_z = x
        
        x = hat_z
        for layer in self.transition:
            if isinstance(layer, SpeciesLinear):
                x = layer(x, species_indices)
            else:
                x = layer(x)
        hat_z_next = x
        
        return hat_z_next, hat_z


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, L, D)"""
        return x + self.pe[:, :x.size(1)]


class SwarmTransformerDecoder(nn.Module):
    """Vectorized Transformer Decoder for interpreting multi-symbol utterances."""
    def __init__(self, num_species, msg_dim, vocab_size, embedding_dim, hidden_dim, output_dim, nhead=4, num_layers=2):
        super(SwarmTransformerDecoder, self).__init__()
        self.msg_dim = msg_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Social attention (who to listen to)
        self.social_attention = SwarmAttention(num_species, msg_dim, vocab_size, hidden_dim, nhead)
        
        # Temporal decoder (interpreting the syntax of the weighted message)
        self.symbol_embedding = nn.Linear(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=msg_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.syntax_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to observation space
        self.output_proj = SpeciesLinear(num_species, embedding_dim * msg_dim, output_dim)
    
    def forward(self, all_msgs, species_indices, mask):
        """
        all_msgs: (N, B, M, V)
        species_indices: (N,)
        mask: (N, N)
        """
        n, b, m, v = all_msgs.shape
        
        # 1. Social Weighting: Determine which agents to listen to.
        # We use the social attention weights to weigh the sequences.
        # msgs_flat for initial social attention
        msgs_flat = all_msgs.view(n, b, -1)
        _, attn_weights = self.social_attention(msgs_flat, species_indices, mask=mask)
        
        # attn_weights: (B, N_query, N_key)
        # Apply weighting to ALL messages (N_key dimension)
        # Reshape messages to (B, N, M*V)
        all_msgs_reshaped = all_msgs.transpose(0, 1).reshape(b, n, -1)
        weighted_msgs_flat = torch.matmul(attn_weights, all_msgs_reshaped) # (B, N, M*V)
        
        # 2. Syntax Decoding: Process the weighted sequence of symbols.
        # Reshape to (B*N, M, V) to treat each agent's summary as a sequence
        weighted_utterance = weighted_msgs_flat.view(b * n, m, v)
        
        # Embed and add positional info
        x = self.symbol_embedding(weighted_utterance)
        x = self.pos_encoding(x)
        
        # Run through Transformer encoder (self-attention over symbol sequence)
        x = self.syntax_transformer(x) # (B*N, M, E)
        
        # 3. Species-specific Projection to Observation
        x = x.view(b, n, -1).transpose(0, 1) # (N, B, M*E)
        recon = self.output_proj(x, species_indices)
        
        return recon, attn_weights


class Swarm(nn.Module):
    """
    Centralized model managing all agents in a vectorized, heterogeneous manner.
    Supports multiple species for Population-Based Training.
    """
    def __init__(self, obs_dim, msg_dim, vocab_size, hidden_dim, latent_dim, 
                 total_agents, action_dim, batch_size, learning_rate, num_species=1, composition_config=None):
        super(Swarm, self).__init__()
        self.total_agents = total_agents
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_species = num_species
        
        # Identity per agent
        self.register_buffer("species_indices", torch.arange(total_agents) % num_species)
        
        # Heterogeneous components
        self.policy_encoder = LSTMEncoder(num_species, obs_dim, hidden_dim, msg_dim, vocab_size)
        self.action_head = ActionHead(num_species, obs_dim, hidden_dim, action_dim)
        
        self.world_model = SwarmWorldModel(num_species, msg_dim, vocab_size, hidden_dim, latent_dim)
        
        # Use Transformer Decoder if configured
        if composition_config and composition_config.get('use_transformer_decoder'):
            self.decoder = SwarmTransformerDecoder(
                num_species, msg_dim, vocab_size, 
                composition_config['embedding_dim'],
                hidden_dim, obs_dim,
                nhead=composition_config['transformer_heads'],
                num_layers=composition_config['transformer_layers']
            )
        else:
            # Fallback to simple decoder if needed (though we currently replaced it)
            # For backward compatibility, we could keep the old one, but I've replaced it in the file.
            # I'll re-add a basic version if necessary, but the Transformer is the Goal.
            self.decoder = SwarmTransformerDecoder(
                num_species, msg_dim, vocab_size, 32, hidden_dim, obs_dim
            )
        
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
        msg_logits, self.hidden_state = self.policy_encoder(obs, self.species_indices, self.hidden_state)
        action_mean = self.action_head(obs, self.species_indices)
        
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
        returns: (N, B)
        target_z: (B, latent_dim)
        mask: (N, N)
        """
        device = obs.device
        
        # 1. SSL Losses
        recon, self.last_attn_weights = self.decoder(msgs_hard, self.species_indices, mask)
        recon_loss = nn.MSELoss()(recon, obs)
        
        hat_z_next, _ = self.world_model(msgs_hard, self.species_indices, mask)
        target_z_exp = target_z.unsqueeze(0).expand(self.total_agents, -1, -1)
        pred_loss = nn.MSELoss()(hat_z_next, target_z_exp)
        
        # Penalty/Regularization
        entropy = -torch.mean(msgs_soft * torch.log(msgs_soft + 1e-10))
        entropy_loss = -entropy_weight * entropy
        comm_cost = comm_cost_weight * torch.mean(torch.abs(msgs_soft))
        
        ssl_loss = recon_loss + pred_loss + entropy_loss + comm_cost
        
        # 2. RL Loss (REINFORCE)
        log_probs = -nn.CrossEntropyLoss(reduction='none')(
            msgs_soft.view(-1, self.vocab_size),
            msgs_hard.argmax(-1).view(-1)
        ).view(self.total_agents, self.batch_size, -1).mean(dim=-1)
        
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)
            
        rl_loss = -torch.mean(log_probs * returns)
        
        total_loss = ssl_loss + rl_loss
        
        # Aggregate stats (weighted by agent performance if needed, but mean is fine for scalar return)
        return total_loss, recon_loss.item(), pred_loss.item()

    def mutate_species(self, target_species_id, source_species_id, mutation_rate=0.01):
        """PBT operator: Copy and mutate parameters between species."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                # We only care about parameters in SpeciesLinear layers that have a species dimension
                if param.dim() >= 2 and param.shape[0] == self.num_species:
                    # Copy
                    param[target_species_id].copy_(param[source_species_id])
                    # Mutate
                    if mutation_rate > 0:
                        noise = torch.randn_like(param[target_species_id]) * mutation_rate
                        param[target_species_id].add_(noise)
        print(f"  [PBT] Mutated Species {target_species_id} from {source_species_id} (rate={mutation_rate})")
