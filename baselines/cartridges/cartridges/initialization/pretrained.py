from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import wandb

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.utils import get_logger

logger = get_logger(__name__)

def _list_cache_files(run_id: str) -> list[str]:
    import wandb
    import re


    api = wandb.Api()

    # Get all files from the run
    files = [file.name for file in api.run(run_id).files()]

    # Filter for cache-*.pt files using regex
    cache_files = [file for file in files if re.match(r"^cache-.*\.pt$", file)]

    # Extract the epoch or step number from each cache file and create a mapping
    file_to_step = {}
    for file in cache_files:
        # Try to match both epoch and step patterns
        match = re.search(r"cache-(epoch|step)(\d+)\.pt", file)
        if match:
            step_num = int(match.group(2))
            file_to_step[file] = step_num

    # Sort the files by their step/epoch number
    sorted_cache_files = sorted(cache_files, key=lambda x: file_to_step.get(x, 0), reverse=True)
    return sorted_cache_files

class KVFromPretrained(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        # path: Path

        wandb_run_id: str
        filename: Optional[str] = None

    def __init__(self, config: Config):
        self.config = config

    def initialize_kv_cache(
        self,
        tokenizer: Optional[AutoTokenizer]=None,
        model: Optional[torch.nn.Module]=None,
        attn_config: Optional[AttnConfig]=None,
    ) -> TrainableCache:
        is_ddp = "LOCAL_RANK" in os.environ
        print(f"is_ddp: {is_ddp}")
        is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)

        logger.info(f"Restoring cache from wandb run {self.config.wandb_run_id}")
        filename = ...

        cache_files = _list_cache_files(self.config.wandb_run_id)
        if len(cache_files) == 0:
            raise ValueError(f"No cache checkpoints found for wandb run {self.config.wandb_run_id}")
        
        if self.config.filename is not None:
            assert self.config.filename in cache_files, f"Cache file {self.config.filename} not found in wandb run {self.config.wandb_run_id}"
            filename = self.config.filename
        else:
            filename = cache_files[0]

        cache_dir = Path(os.environ["CARTRIDGES_OUTPUT_DIR"]) / "checkpoints" / self.config.wandb_run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        path = cache_dir / filename
        if not path.exists():
            logger.info(f"Downloading cache from wandb run {self.config.wandb_run_id} to {cache_dir}")
            if is_rank_zero:
                out = wandb.restore(
                    filename, 
                    run_path=self.config.wandb_run_id, 
                    root=cache_dir,
                )
        if is_ddp:
            dist.barrier()

        logger.info(f"Loading cache from {cache_dir / filename}")
        cache = TrainableCache.from_pretrained(
            str(cache_dir / filename), device='cuda'
        )
                
        return cache
