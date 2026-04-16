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

def evaluate(model_file):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file)
    trainer_class = get_trainer(model_type=config["MODEL_TYPE"], model_name=config["model"])
    trainer_instance = trainer_class(config, model)
    results = trainer_instance.evaluate(test_data, load_best_model=False, show_progress=True)
    print(json.dumps(results, indent=4), flush=True)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default=None, type=Path, required=False, help="Path to the model tensors, defaults to None")
    args = parser.parse_args()
    model_path = args.model_file
    if model_path is None:
        model_path = sorted(list(Path("saved/").glob("GRU4Rec*.pth")), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    evaluate(model_path)