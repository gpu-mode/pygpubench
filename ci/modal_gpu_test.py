"""Run pygpubench GPU tests on a Modal L4 GPU.

Usage: modal run ci/modal_gpu_test.py <wheel_path> <test_dir>
"""

import modal
from pathlib import Path

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)

app = modal.App("pygpubench-ci", image=image)


@app.function(gpu="L4", timeout=600)
def run_tests(whl_bytes: bytes, whl_name: str, test_files: dict[str, bytes]):
    import subprocess
    import sys
    import os

    # Write wheel and install it
    whl_path = f"/tmp/{whl_name}"
    with open(whl_path, "wb") as f:
        f.write(whl_bytes)
    subprocess.run([sys.executable, "-m", "pip", "install", whl_path], check=True)

    # Write test files
    test_dir = "/tmp/tests"
    os.makedirs(test_dir, exist_ok=True)
    for name, content in test_files.items():
        with open(os.path.join(test_dir, name), "wb") as f:
            f.write(content)

    # Run all test scripts
    os.chdir(test_dir)
    failed = []
    for test_file in sorted(test_files):
        if test_file == "submission.py":
            continue
        print(f"\n=== {test_file} ===")
        result = subprocess.run([sys.executable, test_file], text=True)
        if result.returncode != 0:
            failed.append(test_file)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        raise SystemExit(1)


@app.local_entrypoint()
def main(wheel: str, test_dir: str = "test"):
    import glob

    # Read the wheel
    whl_path = Path(wheel)
    whl_bytes = whl_path.read_bytes()

    # Read all test files
    test_path = Path(test_dir)
    test_files = {}
    for f in test_path.glob("*.py"):
        test_files[f.name] = f.read_bytes()

    run_tests.remote(whl_bytes, whl_path.name, test_files)
