# Emergent Symbolic Communication (Vectorized)

An optimized PyTorch implementation of multi-agent emergent communication with discrete symbols, sparse topologies, and distributed world modeling. Refactored for speed and GPU acceleration.

## Goal

Agents develop a discrete "language" to build a shared global world model without direct access to ground truth. Through training, we observe:
- **Symbol specialization**: Vocabulary narrowing and dedicated message meanings.
- **Hub/relay emergence**: Agents learning to attend to "influencers" in the swarm.
- **Temporal grammar**: Sequential structure in message streams.

## Optimized Architecture

The system has been refactored from individual agents to a **Vectorized Swarm** model for maximum throughput:

- **Swarm Model**: A single unified model processing all agents in parallel using 3D tensors `(num_agents, batch_size, dim)`.
- **LSTMEncoder**: Obs → discrete message (Gumbel-Softmax) with agent-specific hidden states.
- **SwarmAttention**: Learned social weighting (4-head) with topology masking.
- **SwarmWorldModel**: Latent state prediction from multi-agent messaging.
- **ActionHead**: Generates forces for swarm dynamics (Cohesion, Separation, Alignment).

## Features

- **High-Speed Vectorization**: Eliminates Python-level agent loops for 5-10x speedup.
- **GPU Native**: Optimized for CUDA execution via `.venv`.
- **External Configuration**: All hyperparameters managed via `config.yaml`.
- **Dynamic Topology**: Curriculum-based sparse connectivity (K-nearest).
- **Emergent Diagnostics**: Automated analysis of vocabulary and attention hubs.
- **LLM Bridge**: Optional interpretation of emergent "language" using local or hub LLMs.

## Quick Start

### 1. Setup Environment
```bash
# Recommended: Use the provided .venv for GPU access
.\.venv\Scripts\activate
pip install torch numpy pyyaml scipy requests
```

### 2. Configure
Modify `config.yaml` to adjust hyperparameters, environment settings, or LLM providers.

### 3. Run Training
```bash
# Standard training
python main.py

# Training with LLM interpretation enabled
python main.py --use-llm

# Specific LLM provider
python main.py --hub --use-llm
```

## Configuration (config.yaml)

- `training`: Batch size, learning rate, eval interval.
- `agents`: Num agents, vocab size, latent/hidden dims.
- `curriculum`: Tau annealing, communication cost ramps.
- `topology`: Sparsity settings (k-neighbors).
- `environment`: Swarm dynamics and oscillatory frequencies.
- `llm`: Provider endpoints and models.

## Output Files

- `training_log.json`: Structured results for all episodes.
- `swarm.pth`: Saved weights for the entire vectorized swarm.

## License

MIT
