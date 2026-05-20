
import os
from pydrantic import BaseConfig, RunConfig
from pydantic import Field

from cartridges.synthesize import SynthesizeConfig

class TokaConfig(BaseConfig):
    model: str
    tokenizer: str | None = None

    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    rope_scaling: str | None = None

    use_hydragen: bool = False
    hydragen_min_group_size: int = 32
    hydragen_min_prefix_len: int = 256

    enable_chosen_logprobs: bool = True
    max_topk_logprobs: int | None = None

    port: int = 10210
    local_proc_name: str = "server"

    log_level: str = "INFO"
    log_procs: list[str] | None = None
    uvicorn_log_level: str = "info"

    stats_report_seconds: float = 5.0
    statsd_server_url: None | str = None
    
    # WandB configuration
    wandb_enabled: bool = False
    wandb_entity: str = "hazy-research"
    wandb_project: str = "tokasaurus"
    wandb_run_name: str | None = None

    page_size: int = 16
    kv_cache_num_tokens: int = 1024 * 128

    torch_compile: bool = False

    # the batch size at which we switch to using async TP
    async_tp_threshold: int | None = None

    max_tokens_per_forward: int = 8192
    max_seqs_per_forward: int = 1024
    prefill_round_up_multiple: int = 16

    scheduling_steps_ahead: int = 8
    stop_string_num_token_lookback: int = 5

    dp_size: int = 1
    pp_size: int = 1
    tp_size: int = 1

    # adding extra stages to hide the latency
    # of sending lm-head results from the end of the pipeline to the start,
    # as well as buffer data dependencies from sequences being rearranged
    # across microbatches (e.g. as sequences finish / new sequences start).
    pp_num_buffer_stages: int = 1

    track_early_stopping: bool = True
    early_stopping_buffer_size: int = 2048
    early_stopping_num_prediction_buckets: int = 1024
    early_stopping_initial_wait: int = 16
    early_stopping_init_mean: float | None = None
    early_stopping_init_std: float | None = None
    max_num_tokens_per_request: int | None = None

    enable_precise_onboard: bool = True
    precise_onboard_batch_size: int = 128
    greedy_prefill: bool = True

    use_spec_allocation: bool = True
    spec_allocation_std_buffer_scale: float = 0.25
    spec_allocation_target_kv_cache_utilization: float = 1.0

    use_cudagraphs: bool = True
    cudagraph_max_size: int = 128
    cudagraph_step: int = 16
    cudagraph_max_kv_indices_per_seq: int = 32768

    # for debugging only, will slow things down
    allocator_sanity_checks: bool = False
    bump_city_population_me: bool = False


class EvaluateTokaConfig(RunConfig):
    synthesize: SynthesizeConfig
    tokasaurus: TokaConfig
    output_dir: str | None = Field(default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."))
    
    # conda environment to run tokasaurus in (optional)
    conda_env: str | None = None

    def run(self):
        evaluate_toka_config(self)


def evaluate_toka_config(
    config: EvaluateTokaConfig,
):
    import subprocess
    import time
    import requests
    from contextlib import contextmanager
    
    @contextmanager
    def tokasaurus_server(toka_config: TokaConfig, conda_env: str | None = None):
        """Context manager to launch and cleanup a tokasaurus server."""
        
        import socket
        
        def is_port_in_use(port: int) -> bool:
            """Check if a port is already in use."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return False
                except OSError:
                    return True
        
        def find_available_port(start_port: int) -> int:
            """Find the next available port starting from start_port."""
            port = start_port
            while is_port_in_use(port):
                print(f"Port {port} is in use, trying {port + 1}")
                port += 1
            return port
        
        # Find an available port
        available_port = find_available_port(toka_config.port)
        if available_port != toka_config.port:
            print(f"Original port {toka_config.port} was in use, using port {available_port}")
        
        # Build the command arguments from TokaConfig
        toka_cmd = ["toka"]
        
        # Special handling for fields that need custom formatting
        special_fields = {
            'kv_cache_num_tokens': f"{toka_config.kv_cache_num_tokens}",
            'log_procs': ','.join(toka_config.log_procs) if toka_config.log_procs else None,
        }
        
        for field_name, field_value in toka_config.__dict__.items():
            # Skip None values for optional fields
            if field_value is None:
                continue
            
            # Use the available port instead of configured port
            if field_name == 'port':
                field_value = available_port
            
            # Use special formatting if defined
            if field_name in special_fields:
                formatted_value = special_fields[field_name]
                if formatted_value is not None:
                    toka_cmd.append(f"{field_name}={formatted_value}")
            else:
                toka_cmd.append(f"{field_name}={field_value}")
        
        # Wrap with conda activation if specified
        if conda_env:
            cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + toka_cmd
        else:
            cmd = toka_cmd
        
        print(f"Starting tokasaurus server with command: {' '.join(cmd)}")
        
        # Start the server - let it output directly to terminal        
        process = subprocess.Popen(cmd)
        
        def ping_server(port: int, timeout_seconds: float = 1.0) -> bool:
            """Check if the tokasaurus server is responsive."""
            try:
                response = requests.get(f"http://localhost:{port}/ping", timeout=timeout_seconds)
                return response.json().get("message") == "pong"
            except requests.RequestException:
                return False
        
        # Wait for server to be ready
        print(f"Waiting for tokasaurus server on port {available_port} to be ready...")
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        while time.time() - start_time < timeout:
            if ping_server(available_port):
                print("Tokasaurus server is ready!")
                break
            time.sleep(2)
        else:
            process.terminate()
            process.wait()
            raise TimeoutError(f"Tokasaurus server failed to start within {timeout} seconds")
        
        try:
            yield f"http://localhost:{available_port}"
        finally:
            print("Shutting down tokasaurus server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't shut down gracefully, killing it...")
                process.kill()
                process.wait()
            print("Tokasaurus server shut down")
    
    # Launch the server and run synthesis
    with tokasaurus_server(config.tokasaurus, config.conda_env) as server_url:
        # Update the synthesize config to use the launched server
        from cartridges.clients.tokasaurus import TokasaurusClient
        
        # Clone the synthesize config and update the client URL
        synthesize_config = config.synthesize.model_copy(deep=True)
        
        # Update the client configuration to point to our launched server
        if hasattr(synthesize_config.synthesizer, 'client'):
            if isinstance(synthesize_config.synthesizer.client, TokasaurusClient.Config):
                synthesize_config.synthesizer.client.url = server_url
                synthesize_config.synthesizer.client.model_name = config.tokasaurus.model
        
        print(f"Running synthesis with server at {server_url}")
        print(config.run_dir)
        synthesize_config.run_dir = config.run_dir
        synthesize_config.run()


    
