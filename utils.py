import torch
import numpy as np
from scipy.stats import pearsonr
import yaml


def load_config(path="config.yaml"):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def compute_returns(rewards, gamma=0.99, device=None):
    """Compute discounted returns for credit assignment."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def compute_grammar_metrics(sequences):
    """
    Compute grammar complexity metrics from sequences of symbols.
    
    Args:
        sequences: list of lists, one per agent
    
    Returns:
        avg_corr: average sequential correlation across agents
        entropy: transition entropy across all sequences
    """
    correlations = []
    for seq in sequences:
        if len(seq) > 1:
            corr, _ = pearsonr(seq[:-1], seq[1:])
            correlations.append(corr)
    avg_corr = np.mean(correlations) if correlations else 0
    
    # Simple entropy for symbol transitions
    transitions = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            key = (seq[i], seq[i+1])
            transitions[key] = transitions.get(key, 0) + 1
    
    if transitions:
        probs = np.array(list(transitions.values())) / sum(transitions.values())
        entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Avoid log(0)
    else:
        entropy = 0
    
    return avg_corr, entropy


def plot_symbol_usage(top_symbols, episode):
    """
    Plot symbol usage as a bar chart.
    
    Args:
        top_symbols: list of (symbol, count) tuples
        episode: current episode number for title
    """
    if not top_symbols:
        return
    
    symbols, counts = zip(*top_symbols)
    plt.figure(figsize=(10, 6))
    plt.bar(symbols, counts)
    plt.xlabel('Symbols')
    plt.ylabel('Usage Count')
    plt.title(f'Symbol Usage at Episode {episode}')
    plt.savefig(f'symbol_usage_ep{episode}.png')
    plt.close()
    print(f"Symbol usage plot saved to symbol_usage_ep{episode}.png")


def visualize_attention(attention_dict, episode):
    """
    Create a directed graph visualization of attention patterns.
    
    Args:
        attention_dict: dict {src_agent: [(tgt_agent, weight), ...]}
        episode: current episode number for title
    """
    G = nx.DiGraph()
    
    for src, targets in attention_dict.items():
        for tgt, weight in targets:
            if weight > 0.2:  # Threshold for strong attends
                G.add_edge(src, tgt, weight=weight)
    
    if G.number_of_edges() == 0:
        print(f"No strong attention edges (>0.2) to visualize at episode {episode}")
        return
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=500, font_size=10)
    plt.title(f'Attention Graph at Episode {episode}')
    plt.savefig(f'attention_graph_ep{episode}.png')
    plt.close()
    print(f"Attention graph saved to attention_graph_ep{episode}.png")
