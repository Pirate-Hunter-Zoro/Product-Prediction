from recbole.quick_start import run_recbole
import argparse
import torch

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    print("Running recbole training...", flush=True)
    run_recbole(
        model=args.model,
        dataset='amazon_m2',
        config_file_list=['scripts/config.yaml']
    )
    
if __name__=="__main__":
    main()