"""Run pygpubench GPU tests on a Modal L4 GPU.

Usage: modal run ci/modal_gpu_test.py
"""

import modal
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "g++-13", "cmake", "ninja-build")
    .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .env({
        "CUDAARCHS": "80;90",
        "CC": "gcc-13",
        "CXX": "g++-13",
        "CMAKE_GENERATOR": "Ninja",
    })
    .add_local_dir(str(_repo / "csrc"), remote_path="/root/pygpubench/csrc")
    .add_local_dir(str(_repo / "python"), remote_path="/root/pygpubench/python")
    .add_local_dir(str(_repo / "test"), remote_path="/root/pygpubench/test")
    .add_local_file(str(_repo / "pyproject.toml"), remote_path="/root/pygpubench/pyproject.toml")
    .add_local_file(str(_repo / "CMakeLists.txt"), remote_path="/root/pygpubench/CMakeLists.txt")
    .add_local_file(str(_repo / "README.md"), remote_path="/root/pygpubench/README.md")
)

app = modal.App("pygpubench-ci", image=image)


@app.function(gpu="L4", timeout=600)
def run_tests():
    import subprocess
    import shutil
    import glob
    import sys
    import os

    # Mounts are read-only; copy to a writable location for the build
    shutil.copytree("/root/pygpubench", "/tmp/pygpubench")
    os.chdir("/tmp/pygpubench")

    subprocess.run(["uv", "build", "--wheel"], check=True)

    whl = glob.glob("dist/*.whl")[0]
    subprocess.run([sys.executable, "-m", "pip", "install", whl], check=True)

    # Run all test scripts in test/
    os.chdir("/tmp/pygpubench/test")
    failed = []
    for test_file in sorted(glob.glob("*.py")):
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
def main():
    run_tests.remote()
