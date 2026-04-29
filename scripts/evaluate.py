import torch
import json
from pathlib import Path
import argparse
import pandas as pd
import copy

from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction
import recbole.quick_start.quick_start as quick_start

from novel_model import NovelModel

# Monkey patch torch.load
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Monkey patch quick_start
_original_get_model = quick_start.get_model
def _patched_get_model(model_name):
    if model_name=="NovelModel":
        return NovelModel
    return _original_get_model(model_name)
quick_start.get_model = _patched_get_model 

def evaluate_per_locale(
    model_file: Path, 
    extra_metrics:list[str]=None, 
    data_dir:Path=Path("data/amazon_m2"),
    locales:tuple[str, ...] = ("uk", "de", "jp", "it", "fr", "es")
    ):
    """Evaluate the given model's performance

    Args:
        model_file (Path): Saved tensors of model
        extra_metrics (list[str], optional): Metric names to union into loaded config's metric list. Defaults to None.
        data_dir (Path, optional): Directory holding the test files by locale. Defaults to Path("data/amazon_m2").
        locales (tuple[str, ...], optional): lowercase locale codes matching file suffixes. Defaults to ("uk", "de", "jp", "it", "fr", "es").
    """
    config, model, dataset, _, _, test_data = load_data_and_model(model_file)
    if extra_metrics is not None:
        metrics = config['metrics']
        metrics = list(set(metrics) | set(extra_metrics))
        config['metrics'] = metrics
    print(f"Metrics: {config['metrics']}", flush=True)
    trainer_class = get_trainer(model_type=config["MODEL_TYPE"], model_name=config["model"])
    trainer_instance = trainer_class(config, model)
    locale_tokens = {}
    for loc in locales:
        loc_df = pd.read_csv(data_dir / f"amazon_m2.test_{loc}.inter", sep="\t")
        ids = loc_df['session_id:token'].astype(str).tolist()
        locale_tokens[loc.upper()] = ids
    locale_ids = {}
    # Obtain set of integer tokens for the session ids in each locale
    for loc, tokens in locale_tokens.items():
        locale_ids[loc] = set(dataset.token2id("session_id", tokens).tolist())
        print(f"Number of sessions in {loc}: {len(locale_ids[loc])}", flush=True)
    results = {}
    orig_interact = test_data._dataset.inter_feat
    session_col = orig_interact['session_id']
    for (locale_code, id_set) in locale_ids.items():
        # Convert to torch tensor the ids of all the sessions in this locale
        id_tensor = torch.tensor(list(id_set), dtype=torch.int64)
        # Find the sessions which belong to this locale
        session_mask = torch.isin(session_col, id_tensor)
        if session_mask.sum().item() != len(id_set):
            raise ValueError(f"Mismatched session count for locale {locale_code}")
    
        # Now for each locale, we want something of this form:
        #     original_interaction.interaction == {                                                                                                 
        #       "session_id":    tensor([12, 47, 88, ..., 3]),       # length 360,625                                                             
        #       "item_id":       tensor([501, 822, 17, ..., 999]),   # length 360,625                                                             
        #       "item_id_list":  tensor([[...], [...], ...]),        # length 360,625 (2-D here)                                                  
        #   }
        # EXCEPT each tensor is filtered down to sessions that belong in this locale
        filtered = {}
        for col_name, col_tensor in orig_interact.interaction.items():
            filtered[col_name] = col_tensor[session_mask]
        
        # Create an interaction that applies to only this locale
        filtered_interaction = Interaction(filtered)
        # A shallow copy of test_dataset_copy is fine because we will be ovewriting a key, and not changing the value associated with said key
        test_dataset_copy = copy.copy(test_data._dataset)
        test_dataset_copy.inter_feat = filtered_interaction
        
        local_subset_loader = FullSortEvalDataLoader(config, test_dataset_copy, sampler=None, shuffle=False)
        
        results[locale_code] = trainer_instance.evaluate(local_subset_loader, load_best_model=False, show_progress=True)
   
    # Evaluate results over all regions
    results["Overall"] = trainer_instance.evaluate(test_data, load_best_model=False, show_progress=True)
    
    # Now evaluate overall for the three regions the paper used
    counts = {
        "UK" : len(locale_ids["UK"]),
        "DE" : len(locale_ids["DE"]),
        "JP" : len(locale_ids["JP"])
    }
    total = sum(counts.values())
    results["Overall (UK/DE/JP)"] = {}
    for metric in results["UK"]:
        results["Overall (UK/DE/JP)"][metric] = 0.0
        for loc, count in counts.items():
            results["Overall (UK/DE/JP)"][metric] += results[loc][metric] * count / total
    
    print(json.dumps(results, indent=4), flush=True)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GRU4Rec", type=str, required=False, help="Name of the model, defaults to GRU4Rec")
    parser.add_argument("--extra-metrics", type=str, nargs="+", default=None, help="space-separated metric names to union into the checkpoint's metric list")
    args = parser.parse_args()
    model_name = args.model
    model_name = sorted(list(Path("saved/").glob(f"{model_name}*.pth")), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    evaluate_per_locale(model_name, args.extra_metrics)