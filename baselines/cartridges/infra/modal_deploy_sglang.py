# sglang_modal_server.py
"""
Deploy a small-scale SGLang server on Modal ― ready for Llama *or* Qwen.

✅  One file, no external helpers.
✅  Explicit, commented steps (image build → volumes → web_server).
✅  Standard SGLang OpenAI-style REST endpoints (`/v1/chat/completions …`).

USAGE
-----
# interactive test, tears down when you quit:
modal run sglang_modal_server.py

# long-running, autoscaling service:
modal deploy sglang_modal_server.py

# point any OpenAI-compatible client to the printed URL, e.g.:
curl $WEB_URL/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"any","messages":[{"role":"user","content":"Hello!"}]}'
"""

import os
import subprocess
import modal
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Configuration knobs (edit or override with env-vars when calling `modal …`)
# ---------------------------------------------------------------------------

MODEL_PATH   = os.environ.get("MODEL_PATH",
                            #   "meta-llama/Llama-3.2-3B-Instruct")          # or "Qwen/Qwen1.5-7B-Chat"
                            "Qwen/Qwen3-8B")
MODEL_REV    = os.environ.get("MODEL_REV",  None)                        # optional HF revision / commit id
GPU_TYPE     = os.environ.get("GPU_TYPE",  "h100")                       # "h100", "a100-80gb", …
GPU_COUNT    = int(os.environ.get("GPU_COUNT", 1))                       # tensor-parallel shards
PORT         = 8000
MINUTES      = 60                                                        # seconds → minutes helper
SGL_VERSION  = "0.4.6.post1"                                             # tested 2025-04-30

# ---------------------------------------------------------------------------
# 2. Build the container image
#    CUDA 12.8 image + Torch 2.5 + SGLang + FlashInfer for cu124 / torch2.5
# ---------------------------------------------------------------------------

BASE_CUDA    = "12.8.0"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{BASE_CUDA}-devel-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git")
    .pip_install(  # add sglang and some Python dependencies
        "transformers",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "starlette==0.41.2",
        "torch==2.7.1",
        "sglang[all]>=0.4.7.post1",
        # as per sglang website: https://sgl-project.github.io/start/install.html
        # extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
        force_build=True,
    )
    )

# Optional: Hugging Face & FlashInfer caches persisted across container restarts
hf_cache_vol       = modal.Volume.from_name("huggingface-cache",  create_if_missing=True)
flashinfer_cache   = modal.Volume.from_name("flashinfer-cache",   create_if_missing=True)

# ---------------------------------------------------------------------------
# 3. Modal app & web-server Function
# ---------------------------------------------------------------------------

app = modal.App(f"sglang-{Path(MODEL_PATH).name.replace('/','-').lower()}-{GPU_COUNT}x{GPU_TYPE}")

@app.function(                                              # one container = one SGLang shard
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    allow_concurrent_inputs=4,    
    min_containers=0,
    max_containers=64,
    # timeout=20 * MINUTES,
    scaledown_window=1 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        # "/root/.cache/flashinfer":  flashinfer_cache,
    },
    secrets=[modal.Secret.from_name("sabri-api-keys")]
)
@modal.web_server(                                          # exposes $WEB_SERVER_URL (printed by Modal)
    port=PORT,
    startup_timeout=10 * MINUTES
)
def serve():
    """
    Starts `sglang.launch_server` **inside** the container.
    The Modal web-server forwards external traffic → this port.
    """
    # Print quick CUDA sanity info to logs
    subprocess.run("nvidia-smi", shell=True, check=False)

    # Assemble SGLang CLI command
    cmd = [
        "python", "-m", "sglang.launch_server",
        f"--model-path={MODEL_PATH}",
        f"--port={PORT}",
        f"--tp-size={GPU_COUNT}",
        f"--host=0.0.0.0",
    ]
    if MODEL_REV:
        cmd.append(f"--revision={MODEL_REV}")

    # Start the server **non-blocking** so Modal can manage the container independently
    print("Launching SGLang with:\n ", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
    print("From Modal script: SGLang server launched successfully!")

# ---------------------------------------------------------------------------
# 4. Local quick-test helper (optional)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(prompt: str = "Explain the moon landings in one sentence."):
    """
    Spins up a temporary container, calls the server once, then exits.
    Useful for smoke-testing locally.
    """
    import requests, json, time

    url = f"http://localhost:{PORT}/v1/chat/completions"   # Modal port-forwarded automatically
    payload = {
        "model": "any",                       # ignored by the backend
        "messages": [{"role": "user", "content": prompt}],
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))
    print(f"\n[served in {time.time() - t0:.1f}s]")