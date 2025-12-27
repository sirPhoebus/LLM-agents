import numpy as np
import torch
import collections
import copy


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


def get_topology_mask(neighbors, num_agents, device):
    """Convert adjacency list to a boolean tensor mask for SwarmAttention."""
    mask = torch.zeros(num_agents, num_agents, device=device)
    for i, nbs in neighbors.items():
        for j in nbs:
            mask[i, j] = 1.0
    return mask


def get_current_k(episode, k_start=8, k_end=2, decay_episodes=3000):
    """
    Curriculum topology: k decays linearly from k_start to k_end.
    Dense at start (learn what to communicate), sparse at end (learn efficiency).
    """
    if episode >= decay_episodes:
        return k_end
    progress = episode / decay_episodes
    k = k_start - (k_start - k_end) * progress
    return max(k_end, int(k))


def survival_selection(agents, fitness_scores, num_elite=3, num_replace=3):
    """
    Evolutionary pressure: Clone top agents, replace bottom agents.
    
    Args:
        agents: List of Agent objects
        fitness_scores: List of fitness scores (higher is better, e.g. -loss)
        num_elite: Number of elite agents to clone
        num_replace: Number of weak agents to replace
    
    Returns:
        Updated agents list, elite_indices, weak_indices
    """
    # Sort by fitness (higher is better)
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    
    elite_indices = sorted_indices[:num_elite]
    weak_indices = sorted_indices[-num_replace:]
    
    # Clone elite agents to replace weak ones
    for i, weak_idx in enumerate(weak_indices):
        elite_idx = elite_indices[i % num_elite]
        # Deep copy the elite agent's state
        agents[weak_idx].load_state_dict(copy.deepcopy(agents[elite_idx].state_dict()))
        # CRITICAL: Also copy the optimizer state to avoid resetting momentum/learning
        agents[weak_idx].optimizer.load_state_dict(copy.deepcopy(agents[elite_idx].optimizer.state_dict()))
    
    return agents, elite_indices, weak_indices
