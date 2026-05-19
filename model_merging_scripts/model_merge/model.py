import json
import os
import pathlib
import re
import time
from glob import glob
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from model_merge.merge import LinearMerger, SlerpMerger, TaskArithmetic, FusionMerger


def _resolve_model_path(model: str) -> str:
    """Return a local path for model, downloading from HuggingFace if needed."""
    if os.path.isdir(model):
        return model
    if re.match(r'^[^/]+/[^/]+$', model):
        from huggingface_hub import snapshot_download
        print(f"Downloading {model} from HuggingFace...")
        return snapshot_download(repo_id=model)
    return model


class Model:
    def __init__(
        self,
        source_models: list[str],
        method: str,
        output_dir: str = None,
        base_model: str = None,
    ):
        self.source_models = [_resolve_model_path(m) for m in source_models]
        if base_model:
            self.base_model = _resolve_model_path(base_model)
        else:
            self.base_model = None
        self.method = method
        self.state_dict = None
        self.metadata = {}
        self.name = self.get_merge_name(self.source_models)
        self.output_path = pathlib.Path(output_dir, self.name).resolve()

        # Load source models during initialization
        self.source_dicts = self.load()
        # Load base model if specified
        self.base_dict = None
        if self.base_model:
            try:
                self.base_dict = self._load_safetensors(self.base_model)
            except Exception as e:
                raise ValueError(
                    f"Failed to load base model from {self.base_model}: {str(e)}"
                )

    def load(self) -> list[Dict[str, torch.Tensor]]:
        """Load all source models and return their state dictionaries"""
        source_dicts = []
        for directory in tqdm(self.source_models, desc="Loading models"):
            try:
                state_dict = self._load_safetensors(directory)
                source_dicts.append(state_dict)
            except Exception as e:
                raise ValueError(f"Failed to load model from {directory}: {str(e)}")
        return source_dicts

    def merge(
        self,
        weights: list[float] = None,
        t: float = None,
        density: list[float] = None,
    ) -> "Model":
        """Merge the loaded state dictionaries using the specified technique"""
        if self.method == "linear":
            merger = LinearMerger(weights=weights, num_models=len(self.source_dicts))
        elif self.method == "slerp":
            merger = SlerpMerger(t=t)
        elif self.method in ("fusion", "fusion_iqr", "fusion_knee"):
            # Choose thresholding method based on variant
            fusion_method = "iqr" if self.method.endswith("_iqr") else "knee"
            merger = FusionMerger(
                method=fusion_method,
                # Reasonable defaults; tweak if you prefer fail-fast behavior:
                rectify_embeddings=True,       # trim 2D mismatches to common submatrix
                on_shape_mismatch="skip",      # for non-2D mismatches, keep base as-is
            )
        elif self.method == "task":
            merger = TaskArithmetic(
                num_models=len(self.source_dicts),
                weights=weights,
                densities=density,
                sparsification_method=None,
                sparsification_rescale=False,
                consensus_method=None,
                normalize_deltas=False,
            )
        elif self.method == "ties":
            merger = TaskArithmetic(
                num_models=len(self.source_dicts),
                weights=weights,
                densities=density,
                sparsification_method="magnitude",
                sparsification_rescale=False,
                consensus_method="sum",
                normalize_deltas=True,
            )
        elif self.method == "dare_ties":
            merger = TaskArithmetic(
                num_models=len(self.source_dicts),
                weights=weights,
                densities=density,
                sparsification_method="random",
                sparsification_rescale=True,
                consensus_method="sum",
                normalize_deltas=False,
            )
        elif self.method == "dare_linear":
            merger = TaskArithmetic(
                num_models=len(self.source_dicts),
                weights=weights,
                densities=density,
                sparsification_method="random",
                sparsification_rescale=True,
                consensus_method=None,
                normalize_deltas=False,
            )
        else:
            raise ValueError(f"Unknown merge method: {self.method}")

        # Merge each tensor individually
        self.state_dict = {}

        # ---- Branch: Fusion (2-models only) ----
        if self.method.startswith("fusion"):
            # Determine base/target dicts
            if self.base_dict is not None:
                # Expect exactly one source (target) when an explicit base is provided
                if len(self.source_dicts) != 1:
                    raise ValueError(
                        "For Fusion with 'base_model' set, provide exactly one source model (the target)."
                    )
                base_dict = self.base_dict
                target_dict = self.source_dicts[0]
            else:
                # No explicit base: use the first two sources as (base, target)
                if len(self.source_dicts) < 2:
                    raise ValueError("Fusion requires two models: base and target.")
                base_dict = self.source_dicts[0]
                target_dict = self.source_dicts[1]

            # Merge using keys from the base; if a key is missing in target, keep base
            for key in tqdm(base_dict.keys(), desc=f"Merging with {self.method}"):
                base_t = base_dict[key]
                tgt_t = target_dict.get(key, None)
                if tgt_t is None:
                    # Target missing this tensor -> keep base tensor
                    self.state_dict[key] = base_t
                    continue
                # Fusion works on a pair [base, target]
                self.state_dict[key] = merger.merge_tensor([base_t, tgt_t])

        # ---- Branch: all other methods (existing behavior) ----
        else:
            base_dict = self.base_dict if self.base_dict else self.source_dicts[0]
            for key in tqdm(base_dict.keys(), desc=f"Merging with {self.method}"):
                tensors = [sd[key] for sd in self.source_dicts]
                self.state_dict[key] = merger.merge_tensor(tensors)

        # Copy metadata from base model or first source model
        self.metadata = (
            (self.base_dict or self.source_dicts[0]).get("metadata", {}).copy()
        )
        return self

    def save(self, max_shard_size: int = 5 * 1024 * 1024 * 1024) -> str:
        """
        Save the merged state dict to disk using safetensors format.

        Returns:
            str: The final directory where the merged model is saved.
        """
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Save the state_dict using the _save_safetensors method
        self._save_safetensors(
            directory=str(self.output_path),
            max_shard_size=max_shard_size,
            metadata=self.metadata,
        )

        # Copy configuration files to the output directory
        self._copy_config_files(str(self.output_path))

        print(f"Merged model saved successfully at: {self.output_path}")
        return str(self.output_path)

    def _load_safetensors(self, directory: str) -> Dict[str, torch.Tensor]:
        """
        Load a model that has been split into multiple safetensors files or a single safetensors file.

        Args:
            directory: Path to directory containing model files

        Returns:
            dict: Combined state dictionary from all parts
        """
        # Check for a single safetensors file
        single_file_path = os.path.join(directory, "model.safetensors")
        if os.path.exists(single_file_path):
            # Load the single safetensors file
            return load_file(single_file_path)

        # Load the index file for multiple parts
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise ValueError(f"Index file not found in {directory}")

        with open(index_path, "r") as f:
            index = json.load(f)

        # Get all safetensors files in directory - Fixed pattern
        pattern = os.path.join(directory, "model-*-of-*.safetensors")
        model_files = sorted(glob(pattern))

        if not model_files:
            raise ValueError(f"No safetensors files found in {directory}")

        # Extract total number of parts from last file name
        last_file = os.path.basename(model_files[-1])
        match = re.search(r"of-(\d+)\.safetensors$", last_file)
        if not match:
            raise ValueError(f"Invalid file naming pattern in {directory}")

        expected_parts = int(match.group(1))
        if len(model_files) != expected_parts:
            raise ValueError(
                f"Expected {expected_parts} model parts, found {len(model_files)}"
            )

        # Load and combine all parts
        combined_state_dict = {}
        for model_file in tqdm(model_files, desc="Loading safetensors"):
            part_dict = load_file(model_file)
            combined_state_dict.update(part_dict)

        # Sort keys to ensure consistent order
        sorted_state_dict = dict(sorted(combined_state_dict.items()))

        # Verify all weights from index are present
        weight_map = index.get("weight_map", {})
        missing_weights = set(weight_map.keys()) - set(sorted_state_dict.keys())
        if missing_weights:
            raise ValueError(f"Missing weights: {missing_weights}")

        return sorted_state_dict

    def _save_safetensors(
        self,
        directory: str,
        max_shard_size: int = 5 * 1024 * 1024 * 1024,
        metadata: Dict[str, Any] = None,
    ):
        """
        Save a state dictionary as split safetensors files.

        Args:
            directory (str): Directory to save files in.
            max_shard_size (int): Maximum size per shard in bytes (default: 5GB).
            metadata (Dict[str, Any], optional): Additional metadata to include.
        """
        os.makedirs(directory, exist_ok=True)

        # Set default metadata if none provided
        if metadata is None:
            metadata = {}

        # Ensure required format metadata is present
        metadata["format"] = "pt"

        # Initialize variables for sharding
        current_shard = {}
        current_shard_size = 0
        shard_index = 1
        weight_map = {}

        # Calculate total size for progress bar
        total_size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.state_dict.values()
        )
        progress_bar = tqdm(
            total=total_size, desc="Saving model parts", unit="B", unit_scale=True
        )

        # Helper function to save current shard
        def save_shard():
            nonlocal shard_index, current_shard
            if current_shard:
                filename = f"model-{shard_index:05d}-of-00000.safetensors"
                filepath = os.path.join(directory, filename)
                save_file(current_shard, filepath, metadata=metadata)
                shard_index += 1
                current_shard = {}

        # Distribute weights across shards
        for key, tensor in self.state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()

            if current_shard_size + tensor_size > max_shard_size:
                save_shard()
                current_shard_size = 0

            current_shard[key] = tensor
            current_shard_size += tensor_size
            weight_map[key] = f"model-{shard_index:05d}-of-00000.safetensors"
            progress_bar.update(tensor_size)

        # Save final shard
        save_shard()
        progress_bar.close()

        # Update filenames with correct total
        total_shards = shard_index - 1

        if total_shards == 1:
            # Single file case: rename to model.safetensors and don't create index
            old_name = "model-00001-of-00000.safetensors"
            new_name = "model.safetensors"
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Saved single model file: {new_name}")
        else:
            # Multiple files case: use original logic with index
            print("Finalizing shard names...")
            for i in tqdm(range(1, total_shards + 1), desc="Renaming shards"):
                old_name = f"model-{i:05d}-of-00000.safetensors"
                new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
                old_path = os.path.join(directory, old_name)
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)

                # Update weight map
                for key in weight_map:
                    if weight_map[key] == old_name:
                        weight_map[key] = new_name

            # Save index file with metadata
            index = {
                "metadata": {"total_size": total_size, **metadata},
                "weight_map": weight_map,
            }
            index_path = os.path.join(directory, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

    def _copy_config_files(self, output_dir: str):
        """Copy all files except *.safetensors files from first source model to output directory"""
        source_dir = self.source_models[0]

        # Get all files in source directory
        all_files = [
            f
            for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f))
        ]

        # Filter out safetensors files and index files
        files_to_copy = [
            f
            for f in all_files
            if not f.endswith(".safetensors") and not f.endswith(".index.json")
        ]

        from shutil import copy2

        for filename in tqdm(files_to_copy, desc="Copying config files"):
            source_path = os.path.join(source_dir, filename)
            copy2(source_path, os.path.join(output_dir, filename))

    def get_merge_name(self, model_dirs: list[str]) -> str:
        """
        Generate default output path based on model directories.
        Format: merge_XB_ID1-ID2-...-IDn_XXXXX_HF where XXXXX is a unique number

        Args:
            model_dirs (list[str]): List of model directory paths.

        Returns:
            str: The default output path.
        """
        # Extract IDs from paths (number before _HF, or trailing numbers, or basename)
        model_ids = []
        for path in model_dirs:
            basename = os.path.basename(os.path.normpath(path))

            # First try the original pattern (number before _HF)
            match = re.search(r".*?(\d+)_HF$", basename)
            if match:
                model_ids.append(match.group(1))
            else:
                # Try to extract trailing numbers
                match = re.search(r"(\d+)$", basename)
                if match:
                    model_ids.append(match.group(1))
                else:
                    # Fall back to using a sanitized version of the basename
                    sanitized = re.sub(r"[^a-zA-Z0-9]", "", basename)[:10]
                    model_ids.append(sanitized if sanitized else "unknown")

        # Try to find model size (XB or XM) from first model path
        size_match = re.search(r"(\d+(?:\.\d+)?)[bBmM]", model_dirs[0])
        model_size = size_match.group(0).upper() if size_match else "unknown"

        # Create unique identifier from timestamp (last 5 digits)
        unique_id = str(time.time_ns())[-5:]

        # Create merged name
        return f"merge_{self.method.replace('_', '')}{model_size}_from{'-'.join(model_ids)}_{unique_id}_HF"
