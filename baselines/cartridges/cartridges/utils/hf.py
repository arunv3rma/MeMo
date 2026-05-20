from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import tempfile
import os

from cartridges.utils import get_logger

if TYPE_CHECKING:
    from cartridges.structs import Conversation


logger = get_logger(__name__)

def read_conversations_from_hf(repo_id: str, token: Optional[str] = None) -> list[Conversation]:
    """Download and read conversations from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        token: HuggingFace token for authentication (if None, uses HF_TOKEN env var)
        
    Returns:
        List of Conversation objects
    """
    from cartridges.structs import read_conversations
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets' to download from HuggingFace Hub")
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    # Download dataset from HuggingFace Hub
    logger.info(f"Downloading dataset from {repo_id}")

    # Try to download all shards by first checking what's available
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
        
        # Find all parquet files in the data directory
        parquet_files = [f for f in repo_files if f.startswith("data/") and f.endswith(".parquet")]
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in data/ directory of {repo_id}")
        
        # Download and combine all shards
        all_conversations = []
        for parquet_file in sorted(parquet_files):
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{parquet_file}"
            conversations = read_conversations(url)
            all_conversations.extend(conversations)
            logger.info(f"Loaded {len(conversations)} conversations from {parquet_file}")
        
        conversations = all_conversations
        
    except ImportError:
        # Fallback to single file if huggingface_hub is not available
        logger.warning("huggingface_hub not available, falling back to single shard")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/train-00000-of-00001.parquet"
        conversations = read_conversations(url)
    
    logger.info(f"Loaded {len(conversations)} conversations from {repo_id}")
    return conversations


def upload_run_dir_to_hf(
    run_dir: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    collection_slug: Optional[str] = None
) -> str:
    """Upload a run directory containing artifacts and config to HuggingFace Hub.
    
    Args:
        run_dir: Path to the run directory containing "artifacts/dataset.pkl/.parquet" and "config.yaml"
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        token: HuggingFace token for authentication (if None, uses HF_TOKEN env var)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        collection_slug: Optional collection slug to add the dataset to (e.g., "username/collection-name")
        
    Returns:
        URL of the uploaded dataset
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
        import yaml
    except ImportError:
        raise ImportError("Please install 'datasets', 'huggingface_hub', and 'pyyaml' to upload to HuggingFace Hub")
    
    run_dir_path = Path(run_dir)
    
    # Check if run directory exists
    if not run_dir_path.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    # Look for dataset file in artifacts directory
    artifacts_dir = run_dir_path / "artifact"
    if not artifacts_dir.exists():
        raise ValueError(f"Artifact directory not found: {artifacts_dir}")
    
    dataset_file = None
    for file_path in artifacts_dir.iterdir():
        if file_path.suffix in [".pkl", ".parquet"]:
            dataset_file = file_path
            break
    
    if dataset_file is None:
        raise ValueError(f"No dataset file (.pkl or .parquet) found in {artifacts_dir}")
    
    # Look for config file
    config_file = run_dir_path / "config.yaml"
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_file}")
    if dataset_file.suffix == ".parquet":
        dataset = Dataset.from_parquet(str(dataset_file))
    else:  # .pkl
        from cartridges.structs import read_conversations
        conversations = read_conversations(str(dataset_file))
        # Convert to temp parquet file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            from cartridges.structs import write_conversations
            write_conversations(conversations, temp_path)
            dataset = Dataset.from_parquet(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("HuggingFace token required. Set HF_TOKEN environment variable or pass token parameter.")
    
    # Upload dataset to HuggingFace Hub
    logger.info(f"Uploading dataset to {repo_id}")
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message or f"Upload run directory dataset from {run_dir}"
    )
    
    # Upload config file as well
    api = HfApi(token=token)
    logger.info(f"Uploading config file to {repo_id}")
    api.upload_file(
        path_or_fileobj=str(config_file),
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"Upload config from {run_dir}"
    )
    
    # Add to collection if specified
    if collection_slug:
        logger.info(f"Adding dataset to collection: {collection_slug}")
        try:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="dataset",
                token=token
            )
            logger.info(f"Successfully added dataset to collection: {collection_slug}")
        except Exception as e:
            logger.warning(f"Failed to add dataset to collection {collection_slug}: {e}")
    
    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Run directory uploaded successfully: {dataset_url}")
    return dataset_url


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload conversations dataset to HuggingFace Hub")
    parser.add_argument("path", help="Path to run directory or individual file (.pkl/.parquet) to upload")
    parser.add_argument("repo_id", help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", help="HuggingFace token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--commit-message", help="Custom commit message")
    parser.add_argument("--collection", help="Collection slug to add the dataset to (e.g., 'username/collection-name')")
    
    args = parser.parse_args()
    
    # Check if path exists
    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {args.path}")
        exit(1)
    
    # Determine if it's a directory or file and upload accordingly
    if path.is_dir():
        logger.info(f"Uploading run directory: {args.path}")
        dataset_url = upload_run_dir_to_hf(
            run_dir=args.path,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            collection_slug=args.collection
        )
    else:
        logger.error(f"Unsupported path: {args.path}. Must be a run directory.")
        exit(1)
    
    print(f"✅ Dataset uploaded successfully: {dataset_url}")