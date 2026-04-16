import torch
import argparse
import recbole
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer


_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load

