# Emergent Symbolic Communication (Vectorized & Evolutionary)

An optimized PyTorch implementation of multi-agent emergent communication with discrete symbols, sparse topologies, and distributed world modeling. Refactored for speed, GPU acceleration, and population-based evolution.

## Goal

Agents develop a structured "language" to build a shared global world model. Through training, we observe:
- **Symbol specialization**: Vocabulary narrowing and dedicated message meanings.
- **Compositional Syntax**: Emerging multi-symbol "utterances" where order and context carry meaning.
- **Social Hierarchies**: Agents learning to attend to "hubs" or "relays" in the swarm.
- **Species Evolution**: Success-based survival and mutation of agent sub-populations.

## Optimized Architecture

The system has been refactored from individual agents to a **Vectorized Swarm** model with evolutionary capabilities:

- **Swarm Model**: Processes all agents in parallel using 3D tensors `(num_agents, batch_size, dim)`.
- **LSTMEncoder**: Obs → sequence of discrete symbols (Gumbel-Softmax) with agent-specific hidden states.
- **SwarmAttention**: Learned social weighting (4-head) with topology masking.
- **SwarmTransformerDecoder**: A syntax-aware decoder that interprets the *order* of symbols in an utterance to reconstruct the world state.
- **ActionHead**: Generates forces for swarm dynamics (Cohesion, Separation, Alignment).

## Advanced Features

- **Differentiable PBT**: Population-Based Training with heterogeneous "species." Elites are selected based on fitness, while underperforming species are replaced and mutated.
- **Hierarchical Syntax**: Transformer-based messaging supports complex, multi-symbol communication patterns.
- **High-Speed Vectorization**: Specialized `SpeciesLinear` layers and batched operations for 5-10x speedup over standard models.
- **Dynamic Topology**: Curriculum-based sparse connectivity (K-nearest) that evolves during training.
- **Emergent Diagnostics**: Automated analysis of vocabulary usage, bigram syntax, and attention hubs.
- **LLM Bridge**: Interpretation of emergent patterns using local (Gemma) or hub (Qwen) LLMs.

## Quick Start

### 1. Setup Environment
```bash
# Recommended: Use the provided .venv for GPU access
.\.venv\Scripts\activate
pip install torch numpy pyyaml scipy requests
```

### 2. Configure
Modify `config.yaml` to adjust hyperparameters, PBT evolution settings, or transformer composition parameters.

### 3. Run Training
```bash
# Standard training (PBT & Syntax enabled by default)
python main.py

# Training with LLM interpretation enabled
python main.py --use-llm
```

## Configuration (config.yaml)

- `training`: Batch size, learning rate, eval interval.
- `agents`: Num agents, vocab size, latent/hidden dims.
- `evolution`: PBT settings (num species, mutation rate, selection interval).
- `composition`: Transformer decoder settings (layers, heads, embedding dim).
- `curriculum`: Tau annealing, entropy decay, communication cost ramps.
- `topology`: Sparsity settings (k-neighbors).
- `environment`: Swarm dynamics and oscillatory frequencies.

## Output Files

- `training_log.json`: Structured results for all episodes.
- `swarm.pth`: Saved weights for the entire vectorized swarm (all species).

## License

MIT
