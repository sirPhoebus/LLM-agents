import torch
import numpy as np


class WorldSimulator:
    def __init__(self, world_dim, latent_dim, num_agents, action_dim, device, num_scouts=2, proximity_weight=1.0):
        self.world_dim = world_dim
        self.latent_dim = latent_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device
        self.num_scouts = num_scouts
        self.proximity_weight = proximity_weight
        
        # Projection matrix from latent to world
        self.projection_mat = torch.randn(latent_dim, world_dim).to(device)
        
        # Project concatenated actions to latent space
        self.action_proj = torch.randn(num_agents * action_dim, latent_dim).to(device)
        
        # Swarm dynamics parameters (loaded from config in main.py or defaults)
        self.cohesion_factor = 0.01
        self.separation_factor = 0.05
        self.noise_factor = 0.05
        self.action_scale = 0.1
        
        # Oscillatory dynamics parameters
        self.oscillation_amplitude = 0.1
        self.oscillation_frequencies = [0.05, 0.08, 0.12]
        self.global_time_step = 0
        self.num_particles = latent_dim // 2
    
    def compute_oscillatory_force(self, positions):
        """Apply periodic forces to swarm particles."""
        batch_size, num_particles, dims = positions.shape
        
        # Multi-frequency oscillation (different particles feel different phases)
        force = torch.zeros_like(positions)
        
        for i, freq in enumerate(self.oscillation_frequencies):
            # Phase offset per particle creates spatial variation
            phase_offset = torch.arange(num_particles, device=positions.device).float() * 0.5
            
            # X-direction force: sin wave
            t = self.global_time_step * freq
            force[:, :, 0] += self.oscillation_amplitude * torch.sin(t + phase_offset)
            
            # Y-direction force: cos wave (90 degrees out of phase)
            force[:, :, 1] += self.oscillation_amplitude * torch.cos(t + phase_offset * 0.7)
        
        # Normalize by number of frequencies
        force = force / len(self.oscillation_frequencies)
        return force
    
    def apply_swarm_dynamics(self, z, aggregated_actions):
        """Apply swarm dynamics to latent state z."""
        positions = z.view(-1, self.num_particles, 2)
        center = positions.mean(dim=1, keepdim=True)
        cohesion = self.cohesion_factor * (center - positions)
        
        diffs = positions.unsqueeze(2) - positions.unsqueeze(1)
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        close_mask = (dists < 1.0).float()
        repulsion = self.separation_factor * (diffs / dists) * close_mask
        separation = repulsion.sum(dim=2) / (close_mask.sum(dim=2) + 1e-6)
        
        # Apply concatenated actions projected to particle space
        if aggregated_actions is not None:
            action_forces = (aggregated_actions @ self.action_proj).view(-1, self.num_particles, 2) * self.action_scale
        else:
            action_forces = torch.zeros_like(positions)
        
        # Apply oscillatory forces (environment-level periodic dynamics)
        oscillatory_force = self.compute_oscillatory_force(positions)
        self.global_time_step += 1  # Increment time
        
        delta = cohesion + separation + oscillatory_force + self.noise_factor * torch.randn_like(positions) + action_forces
        new_positions = positions + delta
        
        # --- Task Reward (Grounded) ---
        # Centroid distance to target
        centroid = new_positions.mean(dim=1) # (B, 2)
        dist_to_target = torch.norm(centroid - self.target_pos, dim=-1) # (B,)
        
        # Progressive reward: closer is better
        task_reward = torch.exp(-dist_to_target / 2.0) * self.proximity_weight
        
        # Original clustering reward
        clustering_reward = -new_positions.var(dim=1).mean(dim=-1)
        
        reward = clustering_reward + task_reward
        
        return new_positions.view(-1, self.latent_dim), reward
    
    def generate_world_batch(self, batch_size, aggregated_actions, z_t=None):
        """Generate world state from latent state."""
        if z_t is None:
            z_t = torch.randn(batch_size, self.latent_dim, device=self.device)
        current_world = z_t @ self.projection_mat
        z_tp1, reward = self.apply_swarm_dynamics(z_t, aggregated_actions)
        target_world = z_tp1 @ self.projection_mat
        return current_world, target_world, z_t, z_tp1, reward
    
    def get_partial_obs(self, world, agent_id, obs_dim):
        """
        Complementary observability + Task grounding.
        First 2 dimensions are reserved for target info (Scouts only).
        """
        # World slice (obs_dim - 2 dimensions)
        real_obs_dim = obs_dim - 2
        start = (agent_id * real_obs_dim) % world.shape[1]
        indices = torch.arange(start, start + real_obs_dim, device=self.device) % world.shape[1]
        world_obs = world.index_select(1, indices)
        
        # Target info (Scout only)
        # We use the swarm centroid relative to target for simplicity, or agent pos
        # Let's say relative to swarm centroid (shared knowledge of scouts)
        if agent_id < self.num_scouts:
            # Scouts see the target vector
            target_info = self.target_pos # (B, 2)
        else:
            # Followers see zeros
            target_info = torch.zeros(world.shape[0], 2, device=self.device)
            
        return torch.cat([target_info, world_obs], dim=1)
    
    def reset_time(self):
        """Reset the global time step for oscillatory forces."""
        self.global_time_step = 0

    def reset(self, batch_size, obs_dim):
        """Initialize environment state."""
        self.reset_time()
        self.last_z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Randomize target position for each batch element
        # Range [-5, 5]
        self.target_pos = (torch.rand(batch_size, 2, device=self.device) - 0.5) * 10.0
        
        world = self.last_z @ self.projection_mat
        obs = torch.stack([self.get_partial_obs(world, i, obs_dim) for i in range(self.num_agents)])
        return obs, self.last_z

    def step(self, actions, obs_dim):
        """
        Advance one step.
        actions: (num_agents, batch_size, action_dim)
        """
        aggregated = actions.transpose(0, 1).reshape(actions.shape[1], -1) # (B, N*A)
        z_tp1, reward = self.apply_swarm_dynamics(self.last_z, aggregated)
        world_next = z_tp1 @ self.projection_mat
        obs_next = torch.stack([self.get_partial_obs(world_next, i, obs_dim) for i in range(self.num_agents)])
        self.last_z = z_tp1
        return obs_next, reward, z_tp1
