import math
from pathlib import Path
from typing import Dict
import argparse
import matplotlib.pyplot as plt

from plot_utils import load_eval_log

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
        pop_data = load_eval_log(pop_path)
        results["Popularity"] = {
            m:pop_data['global']['Overall'][m] for m in METRICS
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

def render_chart(results: Dict[str, Dict[str, float]], output_path: Path):
    """Plot results in bar graph

    Args:
        results (Dict[str, Dict[str, float]]): Performance results
        output_path (Path): Where the graph should be displayed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())
    # squeeze=false keeps return as a 2D array (only matters if there is only one METRIC, but all the same, just being pedantic)
    fig, axes = plt.subplots(nrows=1, ncols=len(METRICS), figsize=(6 * len(METRICS), 5), squeeze=False)
    for i, metric in enumerate(METRICS):
        # Find all values achieved for this metric by each model
        values = [results[model][metric] for model in model_names]
        ax = axes[0, i]
        bars = ax.bar(model_names, values)
        ax.bar_label(bars, fmt="%.4f")
        ax.set_title(metric.upper())
        ax.set_ylabel("score")
    fig.suptitle("Model comparison (Overall scope)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Path to write the rendered PNG (e.g. plots/model_comparison.png")
    args = parser.parse_args()
    
    results = collect_overall_metrics()
    render_chart(results, args.output)

if __name__=="__main__":
    main()