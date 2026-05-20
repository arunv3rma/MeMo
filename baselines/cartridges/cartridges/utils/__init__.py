from typing import Callable, Optional
import requests
import yaml
import logging



def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Borrowed from pytorch lightning"""
    import random
    import numpy as np
    import torch
    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.
    


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)



def get_logger(name: str, **kwargs) -> logging.Logger:
    import sys
    import os

    # Only add handler if we're on rank 0 or LOCAL_RANK is not set
    local_rank = os.environ.get("LOCAL_RANK", "0")
    
    logger = logging.getLogger(f"{name} [rank={local_rank}]", **kwargs)
    logger.setLevel(logging.INFO)

    if local_rank == "0" or True:
        handler = logging.StreamHandler(sys.stdout)  # Send logs to stdout
        handler.setLevel(logging.INFO)  # Set the log level for this handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Customize format
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger



def get_cache_size(model_name: str) -> int:
    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    response = requests.get(url)
    response.raise_for_status()
    config = response.json()
    breakpoint()


def _convert_to_hashable(obj):
    from cartridges.clients.base import Client
    from transformers import AutoTokenizer
    from pydrantic import BaseConfig
    if isinstance(obj, list):
        return tuple(_convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, _convert_to_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, AutoTokenizer):
        return obj.name_or_path
    elif isinstance(obj, Client):
        return obj.config.model_name
    elif isinstance(obj, BaseConfig):
        return _convert_to_hashable(obj.to_dict())
    else:
        return obj

def disk_cache(
    func: Callable = None, *, 
    cache_dir: str,
    force: bool = False
) -> Callable:
    """
    Simple decorator that caches a function's result to disk using pickle.
    The cache file is named based on the function name and arguments.
    Optionally, a custom cache directory can be provided.
    Usage:
        @disk_cache
        def foo(...): ...
    or:
        @disk_cache(cache_dir="my_cache_dir")
        def foo(...): ...
    """
    import functools
    import os
    import pickle
    import hashlib

    def decorator(inner_func):
        os.makedirs(cache_dir, exist_ok=True)

        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            
            hashable_kwargs = _convert_to_hashable(kwargs)
            hashable_args = _convert_to_hashable(args)

            # Create a unique hash for the function and its arguments
            key = (inner_func.__module__, inner_func.__name__, hashable_args, hashable_kwargs)
            key_bytes = pickle.dumps(key)
            hash_digest = hashlib.md5(key_bytes).hexdigest()
            cache_path = os.path.join(cache_dir, f"{inner_func.__name__}_{hash_digest}.pkl")

            if os.path.exists(cache_path) and not force:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            result = inner_func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper

    # Support both @disk_cache and @disk_cache(cache_dir="...")
    if func is not None and callable(func):
        return decorator(func)
    else:
        return decorator