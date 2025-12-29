import os
import yaml

_config = None

def load_config() -> dict:
    """Load configuration from settings.yaml"""
    global _config
    
    if _config is not None:
        return _config
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        "settings.yaml"
    )
    
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    
    return _config


def get(key: str, default=None):
    """Get config value by dot notation (e.g., 'frames.sample_rate')"""
    config = load_config()
    
    keys = key.split(".")
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value