import math
from pathlib import Path
from typing import Dict

from plot_utils import load_eval_log, load_pop_log

SLURM_LOGS_DIR = Path("slurm_logs")
MODELS = ["GRU4Rec", "NARM", "NovelModel"]
DEEP_MODEL_LOGS = {
    model: f"evaluation_{model}_out.txt"\
        for model in MODELS
} 
POP_LOG_FILENAME = "evaluation_Pop_out.txt"
METRICS = ("mrr@100", "recall@100")

def collect_overall_metrics() -> Dict[str, Dict[str, float]]:
    """Return performance metrics of each model

    Returns:
        Dict[str, Dict[str, float]]: Performance metrics of each model
    """
    results ={}
    pop_path = SLURM_LOGS_DIR / POP_LOG_FILENAME
    if pop_path.exists():
        pop_data = load_pop_log(pop_path)
        results["Popularity"] = {
            m:pop_data['global'][m] for m in METRICS
        }
    else:
        results["Popularity"] = {
            m:math.nan for m in METRICS
        }
        
    # Find model metrics
    for model in MODELS:
        deep_path = SLURM_LOGS_DIR / DEEP_MODEL_LOGS[model]
        if deep_path.exists():
            deep_data = load_eval_log(deep_path)
            results[model] = {
                m:deep_data["Overall"][m] for m in METRICS
            }
        else:
            results[model] = {m:math.nan for m in METRICS}
    
    return results