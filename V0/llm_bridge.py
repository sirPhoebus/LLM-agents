"""
LLM Bridge for Emergent Communication System

Connects the multi-agent emergent communication to a local LLM for:
- Symbol interpretation (converting agent symbols to text)
- Collective model analysis (LLM interprets emergent patterns)
- Goal generation (LLM proposes high-level goals for swarm)

Usage:
  --local : Use local LLM at localhost:1234 (default)
  --hub   : Use HuggingFace Hub (Qwen2.5-3B)
"""

import requests
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

# LLM Configuration
LOCAL_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LOCAL_MODEL = "google/gemma-3n-e4b"
HUB_ENDPOINT = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-3B-Instruct"
HUB_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Global provider selection (set via create_llm_client)
_current_provider = "local"


class LLMClient:
    """OpenAI-compatible client for local LLM."""
    
    def __init__(self, endpoint: str = LOCAL_ENDPOINT, model: str = LOCAL_MODEL):
        self.endpoint = endpoint
        self.model = model
        self.provider = "local"
    
    def chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send chat completion request to local LLM."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"LLM request failed: {e}")
            return f"[LLM Error: {e}]"
    
    def test_connection(self) -> bool:
        """Test if LLM is accessible."""
        try:
            response = self.chat([{"role": "user", "content": "Say 'OK' if you can hear me."}], max_tokens=10)
            return "OK" in response or len(response) > 0
        except:
            return False


class HuggingFaceClient:
    """Client for HuggingFace Inference API (Qwen2.5-3B)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.endpoint = HUB_ENDPOINT
        self.model = HUB_MODEL
        self.provider = "hub"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Try to get from environment
            import os
            key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if key:
                self.headers["Authorization"] = f"Bearer {key}"
    
    def chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send chat completion request to HuggingFace."""
        # Format messages for Qwen
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", str(result))
            return str(result)
        except requests.exceptions.RequestException as e:
            print(f"HuggingFace request failed: {e}")
            return f"[HF Error: {e}]"
    
    def test_connection(self) -> bool:
        """Test if HuggingFace API is accessible."""
        try:
            response = self.chat([{"role": "user", "content": "Hi"}], max_tokens=10)
            return len(response) > 0 and "[HF Error" not in response
        except:
            return False


def create_llm_client(provider: str = "local", api_key: Optional[str] = None):
    """
    Factory function to create LLM client.
    
    Args:
        provider: "local" for localhost:1234, "hub" for HuggingFace
        api_key: Optional API key for HuggingFace
    
    Returns:
        LLMClient or HuggingFaceClient instance
    """
    global _current_provider
    _current_provider = provider
    
    if provider == "hub":
        print(f"Using HuggingFace Hub: {HUB_MODEL}")
        return HuggingFaceClient(api_key)
    else:
        print(f"Using Local LLM: {LOCAL_MODEL} at {LOCAL_ENDPOINT}")
        return LLMClient()


def symbols_to_text(symbol_sequences: List[List[int]], vocab_size: int = 64) -> str:
    """
    Convert agent symbol sequences to a text description for LLM.
    
    Args:
        symbol_sequences: List of symbol lists, one per agent
        vocab_size: Total vocabulary size
    
    Returns:
        Text description of the communication patterns
    """
    num_agents = len(symbol_sequences)
    
    # Analyze patterns
    all_symbols = [s for seq in symbol_sequences for s in seq]
    unique_symbols = set(all_symbols)
    from collections import Counter
    freq = Counter(all_symbols)
    top_symbols = freq.most_common(5)
    
    # Build description
    lines = [
        f"## Multi-Agent Communication Analysis",
        f"- {num_agents} agents communicating",
        f"- Vocabulary: {len(unique_symbols)}/{vocab_size} symbols used",
        f"- Top symbols: {[s for s, _ in top_symbols]}",
        "",
        "### Agent Message Sequences:",
    ]
    
    for i, seq in enumerate(symbol_sequences[:5]):  # First 5 agents
        lines.append(f"  Agent {i}: {seq}")
    
    # Detect patterns
    patterns = []
    for i, seq in enumerate(symbol_sequences):
        if len(set(seq)) < len(seq) * 0.6:
            patterns.append(f"Agent {i} shows repetition")
        if len(seq) > 1:
            try:
                corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                if abs(corr) > 0.3:
                    patterns.append(f"Agent {i} has sequential structure (corr={corr:.2f})")
            except:
                pass
    
    if patterns:
        lines.append("\n### Detected Patterns:")
        for p in patterns:
            lines.append(f"  - {p}")
    
    return "\n".join(lines)


def attention_to_text(hub_counts: Dict[int, int], num_agents: int) -> str:
    """
    Convert attention patterns to text description for LLM.
    
    Args:
        hub_counts: Counter of which agents are attended to most
        num_agents: Total number of agents
    """
    if not hub_counts:
        return "Attention is distributed evenly across agents - no clear hubs."
    
    lines = ["## Attention Pattern Analysis"]
    top_hubs = sorted(hub_counts.items(), key=lambda x: -x[1])[:3]
    
    for agent_id, count in top_hubs:
        role = "hub" if count >= num_agents * 0.3 else "relay"
        lines.append(f"- Agent {agent_id}: Attended by {count} agents ({role})")
    
    return "\n".join(lines)


def interpret_emergence(
    llm: LLMClient,
    symbol_sequences: List[List[int]],
    hub_counts: Dict[int, int],
    vocab_used: int,
    vocab_size: int,
    num_agents: int
) -> str:
    """
    Ask LLM to interpret the emergent communication patterns.
    
    Returns: LLM's interpretation as text
    """
    # Build context
    symbols_desc = symbols_to_text(symbol_sequences, vocab_size)
    attention_desc = attention_to_text(hub_counts, num_agents)
    
    prompt = f"""You are analyzing an emergent multi-agent communication system where agents have developed their own symbolic language through training.

{symbols_desc}

{attention_desc}

Vocabulary usage: {vocab_used}/{vocab_size} symbols

Based on these patterns, provide a brief interpretation:
1. What kind of communication structure has emerged?
2. Are there signs of specialization or role differentiation?
3. What does the symbol usage suggest about the agents' "language"?
4. What might the attention patterns indicate about information flow?

Be concise (2-3 sentences per question)."""

    messages = [{"role": "user", "content": prompt}]
    return llm.chat(messages, max_tokens=500, temperature=0.7)


def generate_swarm_goal(
    llm: LLMClient,
    current_state: str,
    reward_history: List[float]
) -> Tuple[str, float]:
    """
    Ask LLM to propose a goal for the swarm based on current state.
    
    Returns: (goal_description, priority_score)
    """
    recent_rewards = reward_history[-10:] if len(reward_history) >= 10 else reward_history
    reward_trend = "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "stable/declining"
    
    prompt = f"""You are guiding a swarm of emergent communication agents.

Current State:
{current_state}

Reward Trend: {reward_trend}
Recent Rewards: {[f'{r:.2f}' for r in recent_rewards[-5:]]}

Propose ONE high-level goal for the swarm to pursue. Format:
GOAL: [brief description]
PRIORITY: [1-10]
RATIONALE: [one sentence]"""

    messages = [{"role": "user", "content": prompt}]
    response = llm.chat(messages, max_tokens=200, temperature=0.8)
    
    # Parse response
    goal = "Continue current behavior"
    priority = 5.0
    
    for line in response.split("\n"):
        if line.startswith("GOAL:"):
            goal = line[5:].strip()
        elif line.startswith("PRIORITY:"):
            try:
                priority = float(line[9:].strip().split()[0])
            except:
                pass
    
    return goal, priority


def goal_to_action_bias(goal: str, action_dim: int = 4) -> np.ndarray:
    """
    Convert a goal description to action biases for agents.
    
    This is a simple heuristic mapping - could be enhanced with embeddings.
    """
    bias = np.zeros(action_dim)
    
    # Simple keyword-based mapping
    goal_lower = goal.lower()
    
    if "cluster" in goal_lower or "gather" in goal_lower or "cohesion" in goal_lower:
        bias[0] = 0.5  # Increase cohesion
    if "spread" in goal_lower or "explore" in goal_lower or "disperse" in goal_lower:
        bias[1] = 0.5  # Increase separation
    if "communicate" in goal_lower or "share" in goal_lower:
        bias[2] = 0.3  # Communication emphasis
    if "coordinate" in goal_lower or "synchronize" in goal_lower:
        bias[3] = 0.3  # Coordination emphasis
    
    return bias


# ============ INTEGRATION FUNCTION ============

def run_llm_interpretation(
    agents,
    neighbors: Dict,
    tau: float,
    device,
    provider: str = "local",
    verbose: bool = True
) -> Optional[Dict]:
    """
    Run LLM interpretation on trained agents.
    
    Args:
        provider: "local" for localhost:1234, "hub" for HuggingFace
    """
    import torch
    from main import BATCH_SIZE, LATENT_DIM, PROJECTION_MAT, get_partial_obs, OBS_DIM, NUM_AGENTS, VOCAB_SIZE
    
    # Initialize LLM with chosen provider
    llm = create_llm_client(provider)
    
    if verbose:
        print("\n" + "="*50)
        print(f"LLM BRIDGE: Testing connection ({provider})...")
    
    if not llm.test_connection():
        print(f"LLM not available ({provider})")
        return None
    
    if verbose:
        print("LLM connected successfully!")
    
    # Collect symbol sequences
    for agent in agents:
        agent.reset_hidden()
    
    z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    symbol_sequences = [[] for _ in range(NUM_AGENTS)]
    
    with torch.no_grad():
        for t in range(5):
            current_world = z_t @ PROJECTION_MAT
            obs_list = [get_partial_obs(current_world, i, OBS_DIM) for i in range(NUM_AGENTS)]
            
            for i in range(NUM_AGENTS):
                msg_hard, _ = agents[i].generate_message(obs_list[i], tau)
                symbol = msg_hard[0, 0].argmax().item()
                symbol_sequences[i].append(symbol)
            
            z_t = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    
    # Collect hub counts from attention weights
    import collections
    hub_counts = collections.Counter()
    for i in range(NUM_AGENTS):
        if agents[i].last_attn_weights is not None:
            avg_weights = agents[i].last_attn_weights.mean(dim=0).cpu().numpy()
            hub_idx = int(np.argmax(avg_weights))
            if avg_weights[hub_idx] > 0.2:
                hub_counts[neighbors[i][hub_idx]] += 1
    
    # Get vocab stats
    all_symbols = [s for seq in symbol_sequences for s in seq]
    vocab_used = len(set(all_symbols))
    
    # Get LLM interpretation
    if verbose:
        print("\nRequesting LLM interpretation...")
    
    interpretation = interpret_emergence(
        llm,
        symbol_sequences,
        dict(hub_counts),
        vocab_used,
        VOCAB_SIZE,
        NUM_AGENTS
    )
    
    if verbose:
        print("\n" + "="*50)
        print("LLM INTERPRETATION:")
        print("="*50)
        print(interpretation)
        print("="*50)
    
    return {
        "interpretation": interpretation,
        "symbol_sequences": symbol_sequences,
        "hub_counts": dict(hub_counts),
        "vocab_used": vocab_used
    }


if __name__ == "__main__":
    # Test LLM connection
    print("Testing LLM Bridge...")
    llm = LLMClient()
    
    if llm.test_connection():
        print("✓ LLM connected!")
        
        # Test interpretation with dummy data
        test_sequences = [[5, 12, 5, 8, 5], [3, 3, 7, 3, 9], [1, 2, 3, 4, 5]]
        test_hubs = {0: 3, 2: 2}
        
        result = interpret_emergence(llm, test_sequences, test_hubs, 15, 64, 10)
        print("\nTest Interpretation:")
        print(result)
    else:
        print("✗ LLM not available. Make sure localhost:1234 is running.")
