import torch
import json
from pathlib import Path
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer
import argparse

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load

def evaluate(model_file: Path, extra_metrics:list[str]=None):
    """Evaluate the given model's performance

    Args:
        model_file (Path): Saved tensors of model
        extra_metrics (list[str], optional): Metric names to union into loaded config's metric list. Defaults to None.
    """
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file)
    if extra_metrics is not None:
        metrics = config['metrics']
        metrics = list(set(metrics) | set(extra_metrics))
        config['metrics'] = metrics
    print(f"Metrics: {config['metrics']}", flush=True)
    trainer_class = get_trainer(model_type=config["MODEL_TYPE"], model_name=config["model"])
    trainer_instance = trainer_class(config, model)
    results = trainer_instance.evaluate(test_data, load_best_model=False, show_progress=True)
    print(json.dumps(results, indent=4), flush=True)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GRU4Rec", type=str, required=False, help="Name of the model, defaults to GRU4Rec")
    parser.add_argument("--extra_metrics", type=str, nargs="+", default=None, help="space-separated metric names to union into the checkpoint's metric list")
    args = parser.parse_args()
    model_name = args.model
    model_name = sorted(list(Path("saved/").glob(f"{model_name}*.pth")), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    evaluate(model_name, args.extra_metrics)