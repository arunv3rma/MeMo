from __future__ import annotations

import os
import time
import textwrap
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import requests

__all__ = ["spin_up_model_server"]

def _wait_for_health(url: str, timeout: int = 300, interval: float = 3.0) -> None:
    """
    Poll a health endpoint until it returns HTTP 200 or timeout is reached.
    Raises TimeoutError on failure.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server {url} did not become healthy in {timeout}s")

def spin_up_model_server(
    config: Dict,
    server_type: str,
    *,
    download_dir: str = "../models",
    log_dir: str = "./logs",
    health_path: str = "/metrics",
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[subprocess.Popen, Path]:
    """
    Start a vLLM server in the background and wait for it to become healthy.

    Parameters
    ----------
    config : dict
        Must contain keys: "device", "model", "port_number", "name", "max_model_len".
    server_type : str
        A label for logs (e.g., "chat_model" or "embedding_model").
    download_dir : str
        Directory where the model repo will be stored (can be relative to caller CWD).
    tensor_parallel_size : int
        vLLM tensor parallel size.
    log_dir : str
        Directory where the server log file will be written.
    health_path : str
        Path to poll for readiness (e.g., "/metrics" or "/v1/models").
    env_overrides : dict | None
        Extra environment variables to pass to the subprocess.

    Returns
    -------
    (proc, log_path) : (subprocess.Popen, pathlib.Path)
        The running process and the path to the log file.

    Raises
    ------
    RuntimeError
        If the process exits immediately after launch.
    TimeoutError
        If the health endpoint never returns 200 within the timeout.
    """
    device = str(config["device"])
    model_id = str(config["model"])
    port = int(config["port_number"])
    served_name = str(config["name"])
    max_model_len = int(config["max_model_len"])
    health_timeout = int(config["health_timeout"])
    tensor_parallel_size = int(config["tensor_parallel_size"])

    # Prepare directories
    download_root = Path(download_dir).resolve()
    local_repo_dir = download_root / model_id
    download_root.mkdir(parents=True, exist_ok=True)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir).resolve() / f"vllm_{server_type}.log"

    try:
        from huggingface_hub import snapshot_download
        local_repo_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_repo_dir),
            repo_type="model",
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise RuntimeError(f"Model download failed via huggingface_hub: {e}")

    script = textwrap.dedent(f"""
        set -euo pipefail
        python3 -m vllm.entrypoints.openai.api_server \
            --host localhost \
            --port {port} \
            --trust-remote-code \
            --model "{local_repo_dir}" \
            --tensor-parallel-size {tensor_parallel_size} \
            --dtype bfloat16 \
            --served-model-name "{served_name}" \
            --max_model_len {max_model_len}
    """)

    # Environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    if env_overrides:
        for k, v in env_overrides.items():
            env[k] = str(v)

    # Launch background process, log to file (overwrite each run; change to "a" to append)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "--noprofile", "--norc", "-c", script],
        cwd=Path(".").resolve(),
        env=env,
        stdout=log_file,
        stderr=log_file,
        text=True,
    )

    print(f"Started {server_type} server with PID {proc.pid}, logging to {log_path}")

    # Small grace period; detect instant crash and surface logs
    time.sleep(2)
    if proc.poll() is not None:
        log_file.flush()
        try:
            last_lines = "".join(Path(log_path).read_text().splitlines(True)[-50:])
        except Exception:
            last_lines = "(could not read logs)"
        raise RuntimeError(
            f"{server_type} server exited immediately with code {proc.returncode}.\n"
            f"See logs: {log_path}\n--- Last log lines ---\n{last_lines}"
        )

    # Wait for health
    _wait_for_health(f"http://localhost:{port}{health_path}", timeout=health_timeout)
    print(f"{server_type} server is up and running")
    return proc, log_path
