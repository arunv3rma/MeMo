# ---
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# You can find a (somewhat out-of-date) video walkthrough of this example and the related scripts on the Modal YouTube channel
# [here](https://www.youtube.com/watch?v=QmY_7ePR1hM).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# To take advantage of optimized kernels for CUDA 12.8, we install PyTorch, flashinfer, and their dependencies
# via an `extra` Python package index.

import json
from pathlib import Path
from typing import Any, Literal

import aiohttp
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# ## Download the model weights

# We'll be running a pretrained foundation model -- Meta's LLaMA 3.1 8B
# in the Instruct variant that's trained to chat and follow instructions.

# Model parameters are often quantized to a lower precision during training
# than they are run at during inference.
# We'll use an eight bit floating point quantization from Neural Magic/Red Hat.
# Native hardware support for FP8 formats in [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core)
# is limited to the latest [Streaming Multiprocessor architectures](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
# like those of Modal's [Hopper H100/H200 and Blackwell B200 GPUs](https://modal.com/blog/announcing-h200-b200).

# You can swap this model out for another by changing the strings below.
# A single B200 GPUs has enough VRAM to store a 70,000,000,000 parameter model,
# like Llama 3.3, in eight bit precision, along with a very large KV cache.

MODEL_NAME = "Qwen/Qwen3-0.6b"
USE_YARN = True
GPU_COUNT = 1
GPU_TYPE: Literal["H100", "H200", "B200", "A100-80GB"] = "H100"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})


# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency at startup in exchange for lowered latency and higher throughput during generation.
# We make this trade-off controllable with the `FAST_BOOT` variable below.
FAST_BOOT = True


app = modal.App(f"vllm-{Path(MODEL_NAME).name.replace('/','-').lower()}-{GPU_COUNT}x{GPU_TYPE}")

MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=60 * MINUTES,  # how long should we stay up with no requests?
    timeout=5 * MINUTES,  # how long should we wait for container start?
    allow_concurrent_inputs=64,   
    min_containers=0,
    max_containers=4,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer":  flashinfer_cache_vol,

    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--pipeline-parallel-size", str(GPU_COUNT)]

    if USE_YARN:
        cmd += ["--rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' --max-model-len 131072"]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)



@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=1 * MINUTES
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```