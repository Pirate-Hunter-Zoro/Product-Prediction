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
    
def load_pop_log(path: Path | str) -> Dict[str, Dict[str, Any]]:
    """Obtain global and session results from the given log file

    Args:
        path (Path | str): Log file containing results

    Returns:
        Dict[str, Dict[str, Any]]: Organized "global" and "session" results
    """
    with open(path, 'r') as f:
        contents = f.read()
        global_half, session_half = contents.split("Session Metrics:")
        return {
            "global": json.loads(global_half[global_half.find('{'): global_half.rfind('}')+1]),
            "session": json.loads(session_half[session_half.find('{'): session_half.rfind('}')+1])
        }