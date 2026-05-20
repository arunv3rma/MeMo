#!/usr/bin/env python3
"""
Backend API server for the dataset visualization app.
Provides endpoints for dataset discovery and loading.
"""

from functools import lru_cache
import os
import pickle
import glob
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import concurrent.futures
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from transformers import AutoTokenizer
from cartridges.structs import read_conversations
import pandas as pd

app = FastAPI(title="Dataset Visualization API", version="1.0.0")

# Configuration from environment variables
CORS_ENABLED = os.getenv('CORS_ENABLED', 'true').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
RELOAD = os.getenv('RELOAD', 'false').lower() == 'true'

# CORS middleware configuration
if CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@lru_cache(maxsize=5)
def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from pickle or parquet file."""
    try:
        # Use the new read_conversations function that handles both formats
        conversations = read_conversations(file_path)
        return conversations
    except ImportError as e:
        print(f"Missing dependency for {file_path}: {e}")
        print("Please install required dependencies: pip install pyarrow pandas")
        return []
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback to old pickle loading for backwards compatibility
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                if 'rows' in data:
                    return data['rows']
                elif 'examples' in data:
                    return data['examples']
                elif 'data' in data:
                    return data['data']
                else:
                    # Try to extract first list value
                    for value in data.values():
                        if isinstance(value, list):
                            return value
                    return []
            elif isinstance(data, list):
                return data
            else:
                return []
        except Exception as e2:
            print(f"Error loading dataset with fallback {file_path}: {e2}")
            return []

def serialize_training_example(example, tokenizer: AutoTokenizer=None, include_logprobs=False) -> Dict[str, Any]:
    """Convert TrainingExample to JSON-serializable format."""
    try:
        messages = []
        for msg in example.messages:
            
            token_ids = msg.token_ids.tolist() if hasattr(msg.token_ids, "tolist") else msg.token_ids

            if token_ids is not None and tokenizer is not None:
                token_strs = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
            else:
                token_strs = None

            message_data = {
                'content': msg.content,
                'role': msg.role,
                'token_ids': token_ids,
                'token_strs': token_strs,
                'top_logprobs': None
            }
            
            # Handle logprobs if they exist and are requested
            if (
                include_logprobs and 
                hasattr(msg, 'top_logprobs') and 
                msg.top_logprobs is not None and 
                token_ids is not None 
            ):
                # Use the original structure to get all top-k alternatives for each position
                top_logprobs_matrix = msg.top_logprobs  # This is the original TopLogprobs object
                
                # Add null checks for the internal arrays before attempting to iterate
                if (top_logprobs_matrix.token_idx is not None and 
                    top_logprobs_matrix.token_id is not None and 
                    top_logprobs_matrix.logprobs is not None):
                    
                    # Create list of lists, same length as token_ids
                    token_idx_to_logprobs = [[] for _ in range(len(token_ids))]
                    
                    for token_idx, token_id, logprobs in zip(top_logprobs_matrix.token_idx, top_logprobs_matrix.token_id, top_logprobs_matrix.logprobs):
                        token_idx_to_logprobs[token_idx].append({
                            'token_id': int(token_id),
                            "token_str": tokenizer.decode([token_id], skip_special_tokens=False) if tokenizer else None,
                            'logprob': float(logprobs)
                        })
                                
                    result = []
                    for token_idx, logprobs in enumerate(token_idx_to_logprobs):
                        # Sort by logprob (highest first)
                        logprobs.sort(key=lambda x: x['logprob'], reverse=True)
                        result.append(logprobs)
                    
                    message_data['top_logprobs'] = result
            
            messages.append(message_data)
        
        # Serialize metadata to handle numpy arrays and other non-serializable objects
        serialized_metadata = {}
        if example.metadata:
            for key, value in example.metadata.items():
                try:
                    # Handle numpy arrays
                    if hasattr(value, 'tolist'):
                        serialized_metadata[key] = value.tolist()
                    # Handle other numpy types
                    elif hasattr(value, 'item'):
                        serialized_metadata[key] = value.item()
                    # Handle regular serializable objects
                    else:
                        # Test if it's JSON serializable
                        import json
                        json.dumps(value)
                        serialized_metadata[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serialized_metadata[key] = str(value)
        
        
        return {
            'messages': messages,
            'system_prompt': example.system_prompt,
            'type': example.type,
            'metadata': serialized_metadata
        }
    except Exception as e:
        print(f"Error serializing example: {e}")
        return {
            'messages': [],
            'system_prompt': '',
            'type': 'unknown',
            'metadata': {}
        }

def quick_check_dataset(file_path: str) -> Optional[int]:
    """Quickly check if a file is a valid dataset and return approximate size."""
    try:
        # For parquet files, we can get row count without loading all data
        if file_path.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                return table.num_rows
            except ImportError:
                print(f"Warning: pyarrow not available, falling back to loading full dataset for {file_path}")
                # Fallback to loading full dataset
                conversations = load_dataset(file_path)
                return len(conversations)
        
        # For pickle files, we need to load to check
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats and get count without loading all examples
        if isinstance(data, dict):
            if 'rows' in data and isinstance(data['rows'], list):
                return len(data['rows'])
            elif 'examples' in data and isinstance(data['examples'], list):
                return len(data['examples'])
            elif 'data' in data and isinstance(data['data'], list):
                return len(data['data'])
            else:
                # Try to find first list value
                for value in data.values():
                    if isinstance(value, list):
                        return len(value)
                return 0
        elif isinstance(data, list):
            return len(data)
        else:
            return 0
    except Exception as e:
        print(f"Error quick-checking dataset {file_path}: {e}")
        return None

@app.get("/api/datasets")
def discover_datasets(output_dir: Optional[str] = Query(None)):
    """Discover and return available datasets without loading full content."""
    
    # If no output_dir specified, try common locations
    search_paths = []
    if output_dir:
        search_paths.append(output_dir)
    
    # Add some common search paths
    search_paths.extend([
        os.path.expanduser('~/code/cartridges/outputs'),
        os.path.expanduser('~/outputs'),
        '/tmp/cartridges_output',
        './outputs'
    ])
    
    # Also check environment variables
    env_output_dir = os.environ.get('CARTRIDGES_OUTPUT_DIR')
    if env_output_dir:
        search_paths.insert(0, env_output_dir)
    
    datasets = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        # Find all .pkl and .parquet files recursively
        pkl_files = glob.glob(os.path.join(search_path, '**/*.pkl'), recursive=True)
        parquet_files = glob.glob(os.path.join(search_path, '**/*.parquet'), recursive=True)
        all_files = pkl_files + parquet_files
        
        for file_path in all_files:
            try:
                # Quick check if it's a valid dataset
                size_bytes = os.path.getsize(file_path)
                size_gb = size_bytes / (1024 ** 3) if size_bytes is not None else None
                if size_gb is not None and size_gb > 0:
                    file_obj = Path(file_path)
                    dataset_name = file_obj.stem
                    
                    # Calculate relative path from search_path
                    try:
                        relative_path = str(file_obj.relative_to(search_path))
                    except ValueError:
                        # If relative_to fails, just use the filename
                        relative_path = file_obj.name
                    
                    datasets.append({
                        'name': dataset_name,
                        'path': file_path,
                        'relative_path': relative_path,
                        'size': size_gb,
                        'directory': str(file_obj.parent)
                    })
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                continue
    
    # Sort datasets by relative path for consistent ordering
    datasets.sort(key=lambda d: d['relative_path'])
    
    return datasets

@app.get("/api/dataset/{dataset_path:path}/info")
def get_dataset_info(dataset_path: str):
    """Get dataset metadata without loading examples."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get total count efficiently
        total_count = quick_check_dataset(dataset_path)
        if total_count is None:
            # Fallback: load dataset to get count
            examples = load_dataset(dataset_path)
            total_count = len(examples)
        
        return {
            'path': dataset_path,
            'total_count': total_count,
            'file_size': os.path.getsize(dataset_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/{dataset_path:path}")
def get_dataset_page(
    dataset_path: str, 
    page: int = Query(0), 
    page_size: int = Query(12),
    search: Optional[str] = Query(None),
    search_messages: Optional[str] = Query('true'),
    search_system_prompt: Optional[str] = Query('false'),
    search_metadata: Optional[str] = Query('false')
):
    """Load and return a specific page of a dataset with optional search."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert search field parameters to booleans
        search_messages_bool = search_messages and search_messages.lower() == 'true'
        search_system_prompt_bool = search_system_prompt and search_system_prompt.lower() == 'true'
        search_metadata_bool = search_metadata and search_metadata.lower() == 'true'
        print(f"Search fields - messages: {search_messages_bool}, system_prompt: {search_system_prompt_bool}, metadata: {search_metadata_bool}")
        
        # Load all examples
        t0 = time.time()
        examples = load_dataset(dataset_path)
        print(f"Loaded dataset in {time.time() - t0} seconds")
        
        # Apply search filter if provided
        if search and search.strip():
            t0 = time.time()
            search_query = search.strip().lower()
            filtered_examples = []
            
            for example in examples:
                matches = []
                
                # Search in message contents (if enabled)
                if search_messages_bool:
                    message_match = any(
                        search_query in msg.content.lower() 
                        for msg in example.messages
                    )
                    matches.append(message_match)
                
                # Search in system prompt (if enabled)
                if search_system_prompt_bool:
                    system_prompt_match = (
                        example.system_prompt and 
                        search_query in example.system_prompt.lower()
                    )
                    matches.append(system_prompt_match)
                
                # Search in metadata (if enabled)
                if search_metadata_bool:
                    metadata_match = False
                    if example.metadata:
                        metadata_match = any(
                            search_query in str(value).lower() 
                            for value in example.metadata.values()
                        )
                    matches.append(metadata_match)
                
                # Include example if any enabled field matches
                if any(matches):
                    filtered_examples.append(example)
            
            examples = filtered_examples
            print(f"Filtered {len(examples)} examples in {time.time() - t0} seconds")
        
        total_count = len(examples)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # Only serialize the requested page
        t0 = time.time()
        page_examples = examples[start_idx:end_idx]
        serialized_examples = []
        for example in page_examples:
            serialized_examples.append(serialize_training_example(example))
        print(f"Serialized examples in {time.time() - t0} seconds")
        
        return {
            'examples': serialized_examples,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size,
            'path': dataset_path,
            'search': search,
            'search_fields': {
                'messages': search_messages_bool,
                'system_prompt': search_system_prompt_bool,
                'metadata': search_metadata_bool
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/example")
def get_dataset_example_with_logprobs(request: Dict[str, Any]):
    """Get a single example with logprobs included."""
    try:
        dataset_path = request.get('dataset_path')

        tokenizer = _get_tokenizer(dataset_path)

        example_index = request.get('example_index')
        
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path is required")
        if example_index is None:
            raise HTTPException(status_code=400, detail="example_index is required")
                
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load the examples
        examples = load_dataset(dataset_path)
        
        if example_index < 0 or example_index >= len(examples):
            raise HTTPException(status_code=404, detail=f"Example index {example_index} not found (dataset has {len(examples)} examples)")
        
        example = examples[example_index]
        serialized_example = serialize_training_example(example, include_logprobs=True, tokenizer=tokenizer)
        
        return {
            'example': serialized_example,
            'index': example_index,
            'total_count': len(examples)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in get_dataset_example_with_logprobs: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    print("Health check called!")
    return {'status': 'healthy'}


def _get_dataset_config(dataset_path: str) -> Dict[str, Any]:
    # Look for config.yaml in the same directory as the dataset
    dataset_dir = os.path.dirname(dataset_path)
    config_path = os.path.join(dataset_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        # Also try the parent directory (common pattern)
        parent_config_path = os.path.join(os.path.dirname(dataset_dir), 'config.yaml')
        if os.path.exists(parent_config_path):
            config_path = parent_config_path
        else:
            return {'config': None, 'path': None}
    
    # Load the YAML config
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return {
        'config': config_data,
        'path': config_path,
        'exists': True
    }

@app.post("/api/dataset/config")
def get_dataset_config(request: Dict[str, Any]):
    """Get the SynthesizeConfig for a dataset if it exists."""
    try:
        dataset_path = request.get('dataset_path')
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path is required")
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return _get_dataset_config(dataset_path)
    
    except Exception as e:
        print(f"Error loading config: {e}")
        return {'config': None, 'path': None, 'exists': False, 'error': str(e)}

@lru_cache(maxsize=3)
def _get_tokenizer(dataset_path: str):
    """Decode token IDs to text using the specified tokenizer."""
    config = _get_dataset_config(dataset_path)["config"]
    tokenizer_name = config["synthesizer"]["client"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer


@app.get("/api/dashboards")
def get_dashboards():
    """Get available dashboards from the registry."""
    try:
        # Import dashboard registry with proper error handling
        import importlib
        import os
        
        # Get absolute path to dashboards directory
        dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
        dashboards_dir = os.path.abspath(dashboards_dir)
        
        print(f"Looking for dashboards in: {dashboards_dir}")
        print(f"Dashboard directory exists: {os.path.exists(dashboards_dir)}")
        print(f"Files in directory: {os.listdir(dashboards_dir) if os.path.exists(dashboards_dir) else 'None'}")
        
        # Import base registry first
        try:
            print(f"Python sys.path: {sys.path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"__file__ directory: {os.path.dirname(__file__)}")
            print(f"Absolute path to dashboards: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboards'))}")
            
            from src.dashboards.base import registry
            print(f"Successfully imported registry, has {len(registry.dashboards)} dashboards")
        except ImportError as e:
            print(f"Failed to import base: {e}")
            return {'error': f'Failed to import base: {str(e)}'}
        
        # Dynamically import all dashboard modules in the directory
        if os.path.exists(dashboards_dir):
            for filename in os.listdir(dashboards_dir):
                if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                    module_name = filename[:-3]  # Remove .py extension
                    print(f"Attempting to import dashboard module: {module_name}")
                    
                    try:
                        # Import the module - this should register any dashboards it contains
                        importlib.import_module(f"src.dashboards.{module_name}")
                        print(f"Successfully imported {module_name}, registry now has {len(registry.dashboards)} dashboards")
                    except ImportError as e:
                        print(f"Here Failed to import {module_name}: {e}")
                        # Continue with other modules even if one fails
                        continue
                    except Exception as e:
                        print(f"Error importing {module_name}: {e}")
                        continue
        else:
            print(f"Dashboard directory does not exist: {dashboards_dir}")
            return {'error': f'Dashboard directory not found: {dashboards_dir}'}
        
        dashboards = []
        for name, dashboard in registry.dashboards.items():
            dashboards.append({
                'name': name,
                'filters': dashboard.filters,
                'table': dashboard.table,
                'score_metric': dashboard.score_metric,
                'step': dashboard.step
            })
        
        print(f"Returning {len(dashboards)} dashboards: {[d['name'] for d in dashboards]}")
        return {'dashboards': dashboards}
    
    except Exception as e:
        import traceback
        error_msg = f"Error in get_dashboards: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

@app.post("/api/dashboard/analyze")
def analyze_run_with_dashboard(request: Dict[str, Any]):
    print(f"Analyzing run with dashboard: {request}")
    """Analyze a W&B run using a specific dashboard."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from src.dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), '..', 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"src.dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry in analyze: {str(e)}") 
            return {'error': f'Failed to import dashboard registry in analyze: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get table specs from dashboard (plots will be loaded separately)
        table_specs = dashboard.tables(run)
        
        # Just get plot metadata, not the actual data
        plot_specs_meta = []
        try:
            # Get plot specs but don't materialize the data yet
            plots_specs = dashboard.plots(run)
            for plot_spec in plots_specs:
                plot_specs_meta.append({
                    'id': plot_spec.id,
                    'plot_name': plot_spec.plot_name,
                    'x_col': plot_spec.x_col,
                    'y_col': plot_spec.y_col,
                    # 'data' will be loaded asynchronously
                })
        except Exception as e:
            print(f"Error getting plot specs: {e}")
            plot_specs_meta = []
        
        # Serialize table specs (metadata only, no data)
        tables = []
        for table_spec in table_specs:
            tables.append({
                'step': table_spec.step,
                'score_col': table_spec.score_col,
                'answer_col': table_spec.answer_col,
                'prompt_col': table_spec.prompt_col,
                'pred_col': table_spec.pred_col,
                'path': table_spec.path,  # Include path for later data loading
                # 'data' will be loaded on-demand via separate endpoint
            })
        
        return {
            'plots': plot_specs_meta,
            'tables': tables,
            'dashboard_name': dashboard_name,
            'run_id': run_id
        }
        
    except Exception as e:
        print(f"Error in analyze_run_with_dashboard: {str(e)}")
        return {'error': str(e)}

@app.post("/api/dashboard/table")
def get_table_data(request: Dict[str, Any]):
    """Load specific table data on demand."""
    try:
        run_id = request.get('run_id')
        table_path = request.get('table_path')
        table_step = request.get('table_step')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, table_path]):
            raise HTTPException(status_code=400, detail="run_id and table_path are required")
        
        # Import wandb and get the run
        import wandb
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Create a TableSpec and materialize it
        from src.dashboards.base import TableSpec
        table_spec = TableSpec(
            run=run,
            path=table_path,
            step=table_step
        )
        
        # Materialize the table data
        df = table_spec.materialize()
        
        # Handle NaN values that can't be JSON serialized
        df = df.fillna('')  # Replace NaN with empty strings
        
        # Convert to records and handle conversation data
        records = df.to_dict('records')
        
        # Process each record to handle conversation data
        import json
        for record in records:
            for key, value in record.items():
                # Check if value is a string that might be JSON representing a conversation
                if isinstance(value, str) and value.strip():
                    try:
                        # Try to parse as JSON
                        if value.startswith('[') or value.startswith('{'):
                            parsed = json.loads(value)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                # Check if it looks like a conversation (list of dicts with role/content)
                                if all(isinstance(item, dict) and 'role' in item and 'content' in item for item in parsed):
                                    record[key] = parsed  # Replace string with parsed conversation
                    except json.JSONDecodeError:
                        # If parsing fails, keep the original string
                        pass
        
        return {
            'data': records,
            'step': table_step,
            'path': table_path
        }
        
    except Exception as e:
        print(f"Error in get_table_data: {str(e)}")
        return {'error': str(e)}

@app.post("/api/dashboard/plots")
def get_plot_data(request: Dict[str, Any]):
    """Load plot data for a dashboard and run."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from src.dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"src.dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry in plots: {str(e)}") 
            return {'error': f'Failed to import dashboard registry in plots: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get and serialize plot specs with data
        plots_specs = dashboard.plots(run)
        plots = []
        for plot_spec in plots_specs:
            # Handle NaN values in plot data
            plot_df = plot_spec.df.fillna('')  # Replace NaN with empty strings
            
            plots.append({
                'id': plot_spec.id,
                'plot_name': plot_spec.plot_name,
                'x_col': plot_spec.x_col,
                'y_col': plot_spec.y_col,
                'data': plot_df.to_dict('records')  # Convert DataFrame to JSON
            })
        
        return {
            'plots': plots,
            'dashboard_name': dashboard_name,
            'run_id': run_id
        }
        
    except Exception as e:
        print(f"Error in get_plot_data: {str(e)}")
        return {'error': str(e)}

@app.post("/api/wandb/runs")
def get_wandb_runs(request: Dict[str, Any]):
    """Fetch W&B runs using the Python API."""
    print("Fetching W&B runs...")
    try:
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        filters = request.get('filters', {})
        tag_filter = filters.get('tag', '').strip()
        run_id_filter = filters.get('run_id', '').strip()
        dashboard_filters = request.get('dashboard_filters', {})
        page = request.get('page', 0)
        per_page = request.get('per_page', 8)
        
        # Try to import wandb
        try:
            import wandb
        except ImportError:
            return {'error': 'wandb package not installed. Please install with: pip install wandb'}
        
        # Initialize wandb API (uses WANDB_API_KEY environment variable)
        try:
            api = wandb.Api()
        except Exception as e:
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        # Fetch runs from the project
        try:
            # Build filters dictionary for W&B API
            wandb_filters = {}
            
            # Apply user-specified filters
            if tag_filter:
                wandb_filters["tags"] = {"$in": [tag_filter]}
            if run_id_filter:
                wandb_filters["name"] = {"$regex": run_id_filter}
            
            # Apply dashboard-specific filters (dict format)
            if isinstance(dashboard_filters, dict):
                for key, value in dashboard_filters.items():
                    if key and value:
                        # Apply dashboard filters directly - they should already be in W&B filter format
                        wandb_filters[key] = value
            
            print(f"Applying filters: {wandb_filters}, page: {page}, per_page: {per_page}")
            
            # Get all runs to check total count
            # all_runs = api.runs(f"{entity}/{project}", filters=wandb_filters, order="-created_at")
            total_count = 0 # len(list(all_runs))
            
            # Get paginated runs
            runs = api.runs(f"{entity}/{project}", per_page=per_page, filters=wandb_filters, order="-created_at")
            paginated_runs = runs
            total_count = len(runs)


            
            # Convert runs to JSON-serializable format
            runs_data = []
            for idx, run in zip(range(per_page * (page + 1)), paginated_runs):
                if idx < per_page * page:
                    continue
                print(f"Processing run {run.id}")
                try:
                    # Safely serialize config and summary to handle complex objects
                    def serialize_dict(d):
                        """Recursively serialize dictionary to handle complex objects."""
                        if not isinstance(d, dict):
                            return str(d) if d is not None else None
                        
                        result = {}
                        for key, value in d.items():
                            if isinstance(value, dict):
                                result[key] = serialize_dict(value)
                            elif isinstance(value, (list, tuple)):
                                result[key] = [serialize_dict(item) if isinstance(item, dict) else str(item) for item in value]
                            elif hasattr(value, '__dict__'):
                                # Convert objects to string representation
                                result[key] = str(value)
                            else:
                                result[key] = value
                        return result
                    
                    run_data = {
                        'id': run.id,
                        'name': run.name or run.id,
                        'state': run.state,
                        'createdAt': run.created_at,
                        'config': serialize_dict(dict(run.config)) if run.config else {},
                        'summary': serialize_dict(dict(run.summary)) if run.summary else {},
                        'tags': list(run.tags) if run.tags else [],
                        'url': run.url
                    }
                    runs_data.append(run_data)
                except Exception as e:
                    print(f"Error processing run {run.id}: {e}")
                    continue
            
            print(f"Fetched {len(runs_data)} runs (page {page}, total available: {total_count})")
            return {
                'runs': runs_data,
                'total': total_count,
                'page': page,
                'per_page': per_page,
                'has_more': (page + 1) * per_page < total_count
            }
            
        except Exception as e:
            return {'error': f'Failed to fetch runs from {entity}/{project}: {str(e)}'}
    
    except Exception as e:
        return {'error': str(e)}

@app.post("/api/dashboard/slices")
def get_table_slices(request: Dict[str, Any]):
    """Get slices for a specific table using dashboard slice functions."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        table_path = request.get('table_path')
        table_step = request.get('table_step')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name, table_path]):
            raise HTTPException(status_code=400, detail="run_id, dashboard_name, and table_path are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from src.dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"src.dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry in slices: {str(e)}")
            return {'error': f'Failed to import dashboard registry in slices: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API and get the run
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Create a TableSpec and materialize the data
        from src.dashboards.base import TableSpec
        table_spec = TableSpec(
            run=run,
            path=table_path,
            step=table_step
        )
        
        # Materialize the table data
        df = table_spec.materialize()
        
        # Get slices from the dashboard
        slices = dashboard.slices(df)
        
        # Convert slices to JSON-serializable format
        slices_data = []
        for slice_obj in slices:
            # Handle NaN values
            slice_df = slice_obj.df.fillna('')
            
            # Convert metrics to JSON-serializable format (handle NaN values)
            serialized_metrics = {}
            for key, value in slice_obj.metrics.items():
                if pd.isna(value):
                    serialized_metrics[key] = None
                else:
                    serialized_metrics[key] = float(value) if isinstance(value, (int, float)) else value
            
            slices_data.append({
                'name': slice_obj.name,
                'data': slice_df.to_dict('records'),
                'count': len(slice_obj.df),
                'metrics': serialized_metrics
            })
        
        return {
            'slices': slices_data,
            'total_count': len(df)
        }
        
    except Exception as e:
        print(f"Error in get_table_slices: {str(e)}")
        return {'error': str(e)}

def process_table_spec_for_slice_metrics(table_spec, dashboard):
    """Process a single table spec to compute slice metrics."""
    try:
        print(f"Computing slice metrics for step {table_spec.step}")
        
        # Materialize the table data
        df = table_spec.materialize()
        
        # Get slices from the dashboard
        slices = dashboard.slices(df)
        
        # Store metrics for each slice at this step
        step_metrics = {'step': table_spec.step}
        slice_data = {}
        
        for slice_obj in slices:
            slice_name = slice_obj.name
            
            # Convert metrics to JSON-serializable format
            serialized_metrics = {}
            for key, value in slice_obj.metrics.items():
                if pd.isna(value):
                    serialized_metrics[key] = None
                else:
                    serialized_metrics[key] = float(value) if isinstance(value, (int, float)) else value
            
            # Store slice data for this step
            slice_data[slice_name] = {
                'step': table_spec.step,
                **serialized_metrics
            }
            
            # Also add to step metrics for easier access
            step_metrics[slice_name] = serialized_metrics
        
        return {
            'step_metrics': step_metrics,
            'slice_data': slice_data,
            'success': True
        }
                
    except Exception as e:
        print(f"Error computing slice metrics for step {table_spec.step}: {e}")
        return {
            'step_metrics': None,
            'slice_data': None,
            'success': False,
            'error': str(e)
        }

@app.post("/api/dashboard/slice-metrics")
def get_slice_metrics_over_time(request: Dict[str, Any]):
    """Compute slice metrics across all table steps for a run and dashboard."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from src.dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"src.dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry in slice-metrics: {str(e)}")
            return {'error': f'Failed to import dashboard registry in slice-metrics: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API and get the run
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get all table specs from dashboard
        table_specs = dashboard.tables(run)
        print(f"Found {len(table_specs)} table specs for slice metrics computation")
        
        # Compute slice metrics for each table step in parallel
        slice_metrics_over_time = {}
        step_data = []
        
        print(f"Starting parallel computation for {len(table_specs)} table specs")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_spec = {
                executor.submit(process_table_spec_for_slice_metrics, table_spec, dashboard): table_spec 
                for table_spec in table_specs
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_spec):
                table_spec = future_to_spec[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        step_metrics = result['step_metrics']
                        slice_data = result['slice_data']
                        
                        # Add to step_data
                        step_data.append(step_metrics)
                        
                        # Process slice data
                        for slice_name, step_data_point in slice_data.items():
                            # Initialize slice tracking if not exists
                            if slice_name not in slice_metrics_over_time:
                                slice_metrics_over_time[slice_name] = {
                                    'name': slice_name,
                                    'data': []
                                }
                            
                            # Add step data to slice
                            slice_metrics_over_time[slice_name]['data'].append(step_data_point)
                    else:
                        print(f"Failed to process step {table_spec.step}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"Error processing result for step {table_spec.step}: {e}")
                    continue
        
        # Sort step_data by step number to maintain order
        step_data.sort(key=lambda x: x['step'])
        
        # Sort slice data by step for each slice
        for slice_name in slice_metrics_over_time:
            slice_metrics_over_time[slice_name]['data'].sort(key=lambda x: x['step'])
        
        print(f"Completed parallel computation for all table specs")
        
        # Convert to list format
        slice_metrics_list = list(slice_metrics_over_time.values())
        
        print(f"Successfully computed slice metrics for {len(slice_metrics_list)} slices across {len(step_data)} steps")
        
        return {
            'slice_metrics': slice_metrics_list,
            'step_count': len(step_data)
        }
        
    except Exception as e:
        print(f"Error in get_slice_metrics_over_time: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Visualization Server')
    parser.add_argument('--host', default=HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=PORT, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=RELOAD, help='Enable auto-reload')
    parser.add_argument('--no-cors', action='store_true', help='Disable CORS')
    parser.add_argument('--cors-origins', default=','.join(CORS_ORIGINS), 
                       help='Comma-separated list of allowed CORS origins')
    
    args = parser.parse_args()
    
    # Override configuration with CLI args
    cors_enabled = not args.no_cors and CORS_ENABLED
    cors_origins = args.cors_origins.split(',') if args.cors_origins != ','.join(CORS_ORIGINS) else CORS_ORIGINS
    
    # Configure CORS middleware
    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"CORS enabled: {cors_enabled}")
    if cors_enabled:
        print(f"CORS origins: {cors_origins}")
    
    print(f"Start the frontend with: VITE_API_TARGET=http://localhost:{args.port} npm run dev")
    print("If you are on a remote machine, you need to forward the port to your local machine and run the frontend on your local machine.")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)