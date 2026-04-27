from recbole.quick_start import run_recbole
import recbole.quick_start.quick_start as quick_start
import argparse
import torch

from novel_model import NovelModel

# Monkey patch the torch.load function
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Monkey patch the quick_start.get_model function
_original_get_model = quick_start.get_model
def _patched_get_model(model_name):
    if model_name == "NovelModel":
        return NovelModel
    return _original_get_model(model_name)
quick_start.get_model = _patched_get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    # Figure out what model we will be using - Recbole is weird so for our novel model we need a direct reference to its class
    # Due to monkey patch of loading model above, this should be redundant but harmless
    model_arg = NovelModel if args.model == "NovelModel" else args.model
    
    print("Running recbole training...", flush=True)
    run_recbole(
        model=model_arg,
        dataset='amazon_m2',
        config_file_list=['scripts/config.yaml']
    )
    
if __name__=="__main__":
    main()