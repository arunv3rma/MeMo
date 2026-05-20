"""
Sets up an SSH server in a Modal container.
This requires you to `pip install sshtunnel` locally.
After running this with `modal run launch_ssh.py`, connect to SSH with `ssh -p 9090 root@localhost`,
or from VSCode/Pycharm.
This uses simple password authentication, but you can store your own key in a modal Secret instead.
"""
import modal
import threading
import socket
import subprocess
import time
import os

MINUTES = os.environ.get("MINUTES", 60)  # seconds
PORT = os.environ.get("PORT", 8000)
GPU_COUNT = os.environ.get("GPU_COUNT", 1)
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
BRANCH = os.environ.get("BRANCH", "main")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "openssh-server", "tmux", "curl")
    .run_commands(
        "mkdir -p /run/sshd",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'root:password' | chpasswd"
    )
    .env({"NVM_DIR": "/root/.nvm"})
    .run_commands(  # install nvm and node for claude code
        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash",
        '. "/root/.nvm/nvm.sh" && nvm install 22',
        '. "/root/.nvm/nvm.sh" && nvm use 22',
        '. "/root/.nvm/nvm.sh" && npm install -g @anthropic-ai/claude-code',
    )
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/ScalingIntelligence/tokasaurus.git /root/tokasaurus",
        secrets=[modal.Secret.from_name("sabri-api-keys")],
    )
    .run_commands(f"cd /root/tokasaurus && git fetch --all")
    .run_commands("cd /root/tokasaurus && pip install -e .[dev]")
)
if BRANCH != "main":
    image = image.run_commands(f"cd /root/tokasaurus && git fetch --all && git checkout --track origin/{BRANCH}")
image = image.run_commands("cd /root/tokasaurus && git pull")




hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
flashinfer_cache_vol = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)


app = modal.App(f"tokasaurus-ssh-{GPU_COUNT}x{GPU_TYPE}")

LOCAL_PORT = 9090

def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
        q.put((host, port))

@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    allow_concurrent_inputs=999,
    scaledown_window=15 * MINUTES,
    secrets=[modal.Secret.from_name("sabri-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    timeout=3600 * 24
)
def launch_ssh(q):

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        subprocess.run(["/usr/sbin/sshd", "-D"])

@app.local_entrypoint()
def main():
    try:
        import sshtunnel
    except ImportError:
        raise ImportError("sshtunnel is not installed. Please install it with `pip install sshtunnel`.")

    with modal.Queue.ephemeral() as q:
        launch_ssh.spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")

        server = sshtunnel.SSHTunnelForwarder(
            (host, port),
            ssh_username="root",
            ssh_password="password",
            remote_bind_address=('127.0.0.1', 22),
            local_bind_address=('127.0.0.1', LOCAL_PORT),
            allow_agent=False
        )

        try:
            server.start()
            print(f"SSH tunnel forwarded to localhost:{server.local_bind_port}")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SSH tunnel...")
        finally:
            server.stop()