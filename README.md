# Emergent Symbolic Communication

A PyTorch implementation of multi-agent emergent communication with discrete symbols, sparse topologies, and distributed world modeling.

## Goal

Agents develop a discrete "language" to build a shared global world model without direct access to ground truth. Through training, we observe:
- Symbol specialization (vocabulary narrowing)
- Hub/relay emergence (attention patterns)
- Temporal grammar (LSTM message sequences)

## Architecture

  MULTI-AGENT SYSTEM (10 agents)
  ├── LSTMEncoder: obs → discrete message (Gumbel-Softmax)
  ├── AttentionDecoder: messages → reconstructed obs
  ├── MessageAttention: learned neighbor weighting (4-head)
  ├── ActionHead: obs → action (affects swarm dynamics)
  └── WorldEstimator: messages → latent state prediction

  Communication: k-nearest topology (sparse, routing needed)
  World: Swarm dynamics with cohesion/separation/actions

## Features

- Discrete Communication: Gumbel-Softmax for differentiable discrete messages
- Sparse Topology: Ring, k-nearest, or full connectivity
- LSTM Memory: Temporal context across timesteps
- Multi-head Attention: Learned message aggregation (who to listen to)
- Communication Cost: L1 penalty encourages concise messages
- Cooperative Reward: Swarm clustering incentive
- Multi-step Rollouts: 5 timesteps per episode for negotiation

## Quick Start

  # Install dependencies
  pip install torch numpy matplotlib

  # Run training
  python main.py

## GPU Optimization (RTX 4090)

- torch.compile() for JIT compilation
- Mixed precision (AMP) with GradScaler
- TF32 matmuls enabled
- Batch size = 256

## Emergence Diagnostics

After training, the evaluation prints:
1. Attention Patterns - Identifies hub/relay agents
2. Symbol Specialization - Vocab usage narrowing
3. Temporal Patterns - LSTM grammar detection

## Key Hyperparameters

  NUM_AGENTS = 10
  TOPOLOGY = 'knearest'
  K_NEIGHBORS = 4
  VOCAB_SIZE = 64
  MSG_DIM = 5
  TIMESTEPS_PER_EPISODE = 5
  LEARNING_RATE = 0.001
  MAX_GRAD_NORM = 1.0

## Training Signals

Watch for these emergence indicators:
- Recon/Pred loss decreasing
- Vocab usage < 70% (specialization)
- Attention weight > 0.2 to specific neighbors (hubs)
- Sequential correlation in message sequences (grammar)

## Output Files

- topology.png: Communication graph visualization
- training_progress.png: Loss curves
- agent_*.pth: Saved model weights

## Future Directions

- Curriculum topology (dense → sparse)
- Survival selection (top-k reproduce)
- Larger agent populations (50+)
- Compositional message analysis

## License

MIT
