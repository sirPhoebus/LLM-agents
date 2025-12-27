import json
import time
from datetime import datetime
import numpy as np


class TrainingLogger:
    """Logs training metrics and emergence signals to JSON file."""
    
    def __init__(self, log_file: str = "training_log.json"):
        self.log_file = log_file
        self.start_time = time.time()
        self.data = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "hyperparameters": {},
            "episodes": [],
            "evaluations": [],
            "final_metrics": {}
        }
    
    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters at start of run."""
        self.data["hyperparameters"] = kwargs
    
    def log_episode(self, episode: int, recon_loss: float, pred_loss: float, 
                    reward_loss: float, tau: float, vocab_used: int = None):
        """Log metrics for an episode."""
        self.data["episodes"].append({
            "episode": episode,
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "reward_loss": reward_loss,
            "tau": tau,
            "vocab_used": vocab_used,
            "elapsed_sec": time.time() - self.start_time
        })
    
    def log_evaluation(self, episode: int, eval_data: dict):
        """Log periodic evaluation results."""
        self.data["evaluations"].append({
            "episode": episode,
            **eval_data
        })
    
    def log_final(self, final_metrics: dict):
        """Log final run metrics."""
        self.data["final_metrics"] = final_metrics
        self.data["end_time"] = datetime.now().isoformat()
        self.data["total_duration_sec"] = time.time() - self.start_time
    
    def save(self):
        """Save log to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"Training log saved to {self.log_file}")
    
    def get_data(self) -> dict:
        """Get log data for analysis."""
        return self.data


def analyze_run(log_file: str = "training_log.json"):
    """
    Analyze a training run from the log file and print summary.
    """
    with open(log_file, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading log file: {e}")
            return
    
    episodes = data.get("episodes", [])
    evaluations = data.get("evaluations", [])
    hp = data.get("hyperparameters", {})
    final = data.get("final_metrics", {})
    
    print("\n" + "="*60)
    print("TRAINING RUN ANALYSIS")
    print("="*60)
    print(f"Run ID: {data.get('run_id', 'unknown')}")
    print(f"Duration: {data.get('total_duration_sec', 0)/60:.1f} minutes")
    
    total_eps = episodes[-1]['episode'] + 1 if episodes else 0
    print(f"Episodes: {total_eps} ({len(episodes)} data points)")
    
    # Hyperparameters
    print("\n--- Hyperparameters ---")
    for key, val in hp.items():
        print(f"  {key}: {val}")
    
    # Loss trajectory
    if episodes:
        first_10 = episodes[:10]
        last_10 = episodes[-10:]
        
        avg_recon_start = np.mean([e["recon_loss"] for e in first_10])
        avg_recon_end = np.mean([e["recon_loss"] for e in last_10])
        avg_pred_start = np.mean([e["pred_loss"] for e in first_10])
        avg_pred_end = np.mean([e["pred_loss"] for e in last_10])
        
        print("\n--- Loss Trajectory ---")
        print(f"  Recon Loss: {avg_recon_start:.4f} -> {avg_recon_end:.4f} ({(1-avg_recon_end/avg_recon_start)*100:.1f}% reduction)")
        print(f"  Pred Loss:  {avg_pred_start:.4f} -> {avg_pred_end:.4f} ({(1-avg_pred_end/avg_pred_start)*100:.1f}% reduction)")
    
    # Emergence signals
    print("\n--- Emergence Signals ---")
    if evaluations:
        last_eval = evaluations[-1]
        vocab_used = last_eval.get("vocab_used", "N/A")
        hub_counts = last_eval.get("hub_counts", {})
        vocab_total = hp.get("vocab_size", 64)  # Pull from hyperparams, fallback to 64
        
        print(f"  Vocab Used: {vocab_used}/{vocab_total}")
        if hub_counts:
            top_hub = max(hub_counts.items(), key=lambda x: x[1]) if hub_counts else None
            print(f"  Top Hub: Agent {top_hub[0]} (attended by {top_hub[1]} agents)" if top_hub else "  No clear hubs")
        
        # Check for temporal patterns
        temporal_patterns = last_eval.get("temporal_patterns", [])
        if temporal_patterns:
            print(f"  Temporal Patterns: {len(temporal_patterns)} detected")
    else:
        print("  No evaluations logged")
    
    # Final verdict
    print("\n--- Emergence Assessment ---")
    if evaluations:
        last_eval = evaluations[-1]
        vocab = last_eval.get("vocab_used", 64)
        hubs = len(last_eval.get("hub_counts", {}))
        vocab_total = hp.get("vocab_size", 64)  # Pull from hyperparams, fallback to 64
        
        emergence_score = 0
        if vocab < 0.7 * vocab_total:
            emergence_score += 2
            print("  ✓ Symbol specialization (vocab < 70%)")
        if vocab < 0.5 * vocab_total:
            emergence_score += 1
            print("  ✓ Strong specialization (vocab < 50%)")
        if hubs > 0:
            emergence_score += 2
            print("  ✓ Hub/relay patterns detected")
        if 'avg_recon_end' in locals() and avg_recon_end < avg_recon_start * 0.5:
            emergence_score += 1
            print("  ✓ Significant loss reduction")
        
        print(f"\n  EMERGENCE SCORE: {emergence_score}/6")
        if emergence_score >= 4:
            print("  -> Strong signs of emergence!")
        elif emergence_score >= 2:
            print("  -> Weak emergence, needs more training")
        else:
            print("  -> No clear emergence yet")
    
    print("="*60)
    return data
