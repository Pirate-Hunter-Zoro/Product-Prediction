import math
from pathlib import Path
from typing import Dict
import numpy as np
import argparse
import matplotlib.pyplot as plt

from plot_utils import load_eval_log

SLURM_LOGS_DIR = Path("slurm_logs")
MODELS = ["GRU4Rec", "NARM", "NovelModel"]
DEEP_MODEL_LOGS = {model: f"evaluation_{model}_out.txt" for model in MODELS}
POP_LOG_FILENAME = "evaluation_Pop_out.txt"
METRICS = ("mrr@100", "recall@100")
LOCALES = ("UK", "DE", "JP", "IT", "FR", "ES")

def collect_per_locale_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Read slurm logs to obtain per-locale results

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Model performance results over different locales
    """
    results = {}
    pop_path = SLURM_LOGS_DIR / POP_LOG_FILENAME
    if pop_path.exists():
        pop_data = load_eval_log(pop_path)
        results["Popularity"] = {
            locale: {m: pop_data["global"][locale][m] for m in METRICS}
            for locale in LOCALES
        }
    else:
        # Pop log missing
        results["Popularity"] = {locale: {m: math.nan for m in METRICS} for locale in LOCALES}
        
    # Loop over the deep models
    for model in MODELS:
        deep_path = SLURM_LOGS_DIR / DEEP_MODEL_LOGS[model]
        if deep_path.exists():
            deep_data = load_eval_log(deep_path)
            results[model] = {locale : {m: deep_data[locale][m] for m in METRICS} for locale in LOCALES}
        else:
            # Model log missing
            results[model] = {locale: {m: math.nan for m in METRICS} for locale in LOCALES}
    
    # Return locale results of all models
    return results

def render_per_locale_chart(results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path):
    """Create plot for each model of results per locale

    Args:
        results (Dict[str, Dict[str, Dict[str, float]]]): Model metric results by the locale
        output_path (Path): Where to save the plot to
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())
    n_models = len(model_names)
    x = np.arange(len(LOCALES)) # One integer anchor per locale
    bar_width = 0.8 / n_models
    
    fig, axes = plt.subplots(nrows=1, ncols=len(METRICS), figsize=(7 * len(METRICS), 5), squeeze=False)
    for i, metric in enumerate(METRICS):
        ax = axes[0, i]
        for j, model in enumerate(model_names):
            # e.g. n_models = 4, offsets are -1.5*w, -0.5*w, +0.5*w, +1.5*w
            offset = (j - (n_models - 1) / 2) * bar_width # Symmetric anchoring
            values = [results[model][locale][metric] for locale in LOCALES]
            # Draw bars for each locale score
            ax.bar(x + offset, values, bar_width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(LOCALES)
        ax.set_title(metric.upper())
        ax.set_ylabel("score")
        ax.legend()
    fig.suptitle("Per-locale model performances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Path to write the rendered PNG (e.g. plots/per_locale.png)")
    args = parser.parse_args()
    results = collect_per_locale_metrics()
    render_per_locale_chart(results, args.output)

if __name__=="__main__":
    main()