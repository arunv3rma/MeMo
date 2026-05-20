import json
import time
from typing import List, Optional, Literal
import uuid
import modal
import modal.experimental
from pathlib import Path
import socket

from pydantic import BaseModel
import requests
import threading



root = Path(__file__).parent.parent.parent

BRANCH = "main"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/tokasaurus.git /root/tokasaurus",
    )
    .run_commands("cd /root/tokasaurus && pip install -e .")
    .run_commands("pip install --upgrade transformers")
)
if BRANCH != "main":
    image = image.run_commands(f"cd /root/tokasaurus && git fetch --all && git checkout --track origin/{BRANCH}")
image = image.run_commands("cd /root/tokasaurus && git pull", force_build=True)


hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)

MINUTES = 60  # seconds
PORT = 8080

MODEL_NAME = "Qwen/Qwen3-4B"
DP_SIZE = 1
PP_SIZE = 1
MAX_TOPK_LOGPROBS = 20
GPU_TYPE: Literal["H100", "H200", "B200", "A100-80GB"] = "A100-80GB"
MIN_CONTAINERS = 1
MAX_CONTAINERS = 24
ALLOW_CONCURRENT_INPUTS = 2


gpu_count = DP_SIZE * PP_SIZE
model_short = MODEL_NAME.lower().split("/")[-1]
app = modal.App(f"toka-{model_short}-{gpu_count}x{GPU_TYPE}-min{MIN_CONTAINERS}")


def wait_for_port(port, host="localhost", timeout=60.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Port {port} on {host} not ready after {timeout} seconds")


@app.cls(
    image=image,
    gpu=f"{GPU_TYPE}:{DP_SIZE}",
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    scaledown_window=5 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    experimental_options={"flash": "us-west"},
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
)
class Serve:
    @modal.enter()
    def enter(self):
        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()

        wait_for_port(PORT)
        self.flash_manager = modal.experimental.flash_forward(PORT)

    @modal.exit()
    def exit(self):
        self.flash_manager.stop()


def serve():
    import subprocess
    import os
    from fastapi import FastAPI, Request
    from openai import OpenAI

    os.system("nvidia-smi")
    os.system("which nvidia-smi")
    os.environ["OPENAI_API_KEY"] = "*"     # placeholder

    PING_TIMEOUT_SECONDS = 1.0
    WAIT_FOR_SERVER_BACKOFF_SECONDS = 1.0
    
    cmd = [
        "toka",
        f"model={MODEL_NAME}",
        f"kv_cache_num_tokens='({400_000})'",
        f"max_seqs_per_forward={1024}",
        f"max_topk_logprobs={MAX_TOPK_LOGPROBS}",
        f"port={PORT}",
        f"dp_size={DP_SIZE}",
    ]
    def ping() -> bool:
        """Check if the server is responsive. ReturnsTrue if the server responds with 
        "pong", False otherwise.
        """
        try:
            response = requests.get(
                f"http://localhost:{PORT}/ping", timeout=PING_TIMEOUT_SECONDS
            )
            return response.json()["message"] == "pong"
        except requests.RequestException:
            return False

    TIMEOUT = 300
    
    subprocess.Popen(" ".join(cmd),shell=True)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if ping():
            break
        time.sleep(WAIT_FOR_SERVER_BACKOFF_SECONDS)
