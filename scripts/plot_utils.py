import json
from pathlib import Path
from typing import Dict, Any

def load_eval_log(path: Path | str) -> Dict[str, Any]:
    """Return JSON evaluations for the model associated with the output path

    Args:
        path (Path | str): Output log to parse results from

    Returns:
        Dict[str, Any]: Parsed results
    """
    with open(path, 'r') as f:
        contents = f.read()
        json_string = contents[contents.find('{'):contents.rfind('}')+1]
        return json.loads(json_string) 