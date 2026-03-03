import dataclasses
import math
import multiprocessing
import multiprocessing as mp
import os
import traceback
import secrets

from typing import Optional

from . import _pygpubench
from ._types import *
from .utils import DeterministicContext

__all__ = [
    "do_bench_impl",
    "do_bench_isolated",
    "basic_stats",
    "BenchmarkResult",
    "BenchmarkStats",
    "DeterministicContext",
    "KernelFunction",
    "TestGeneratorInterface",
    "ExpectedResult",
]


def do_bench_impl(out_fd: "multiprocessing.Pipe", signature: "multiprocessing.Pipe", qualname: str, test_generator: TestGeneratorInterface,
                  test_args: dict, repeats: int, seed: int, stream: int = None, discard: bool = True,
                  nvtx: bool = False, tb_conn: "multiprocessing.Pipe" = None):
    """
    Benchmarks the kernel referred to by `qualname` against the test case returned by `test_generator`.
    :param out_fd: Writable file descriptor to which benchmark results are written.
    :param signature: Authentication token read by the C++ layer before untrusted code runs.
    :param qualname: Fully qualified name of the kernel object, e.g. ``my_package.my_module.kernel``.
    :param test_generator: A function that takes the test arguments (including a seed) and returns a test case; i.e., a tuple of (input, expected)
    :param test_args: keyword arguments to be passed to `test_generator`. Seed will be generated automatically.
    :param repeats: Number of times to repeat the benchmark. `test_generator` will be called `repeats` times.
    :param stream: Cuda stream on which to run the benchmark. If not given, torch's current stream is selected.
    :param discard: If true, then cache lines are discarded as part of cache clearing before each benchmark run.
    :param nvtx: Whether to enable NVTX markers for the benchmark. Mostly useful for debugging.
    :param tb_conn: A connection to a multiprocessing pipe for sending tracebacks to the parent process.
    """
    assert repeats > 1
    if stream is None:
        import torch
        stream = torch.cuda.current_stream().cuda_stream

    try:
        with DeterministicContext():
            _pygpubench.do_bench(
                out_fd.fileno(),
                signature.fileno(),
                qualname,
                test_generator,
                test_args,
                repeats,
                seed,
                stream,
                discard,
                nvtx,
            )
    except BaseException:
        if tb_conn is not None:
            tb_conn.send(traceback.format_exc())
        raise


@dataclasses.dataclass
class BenchmarkResult:
    event_overhead_us: float
    time_us: list[float]
    errors: Optional[int]
    full: bool = False


@dataclasses.dataclass
class BenchmarkStats:
    """Summary statistics for a microbenchmark run.

    Attributes:
        runs:     Number of timed iterations.
        expected: Expected number of iterations.
        best:     Fastest observed time (µs).
        worst:    Slowest observed time (µs).
        median:   Median time (µs).
        mean:     Arithmetic mean time (µs).
        std:      Sample standard deviation (µs).
        err:      Standard error of the mean (µs), i.e. std / sqrt(runs).
    """

    runs: int
    expected: int
    best: float         # aka fastest
    worst: float        # aka slowest
    median: float
    mean: float
    std: float
    err: float

    def __str__(self):
        if self.runs != self.expected:
            return f"⚠️ {self.mean:.1f} ± {self.std:.2f} µs [{self.best:.1f} - {self.median:.1f} - {self.worst:.1f}]"
        else:
            return f"{self.mean:.1f} ± {self.std:.2f} µs [{self.best:.1f} - {self.median:.1f} - {self.worst:.1f}]"


def basic_stats(time_us: list[float]) -> BenchmarkStats:
    valid = [t for t in time_us if t > 0]
    runs = len(valid)
    fastest = min(valid)
    slowest = max(valid)
    median = sorted(valid)[runs // 2]
    mean = sum(valid) / runs
    variance = sum(map(lambda x: (x - mean)**2, valid)) / (runs - 1)
    std = math.sqrt(variance)
    err = std / math.sqrt(runs)

    return BenchmarkStats(runs, len(time_us), fastest, slowest, median, mean, std, err)


def do_bench_isolated(
        qualname: str,
        test_generator: TestGeneratorInterface,
        test_args: dict,
        repeats: int,
        seed: int,
        *,
        discard: bool = True,
        nvtx: bool = False,
        timeout: int = 300,
) -> BenchmarkResult:
    """
    Runs kernel benchmark (`do_bench_impl`) in a subprocess for proper isolation.
    """
    assert repeats > 1

    _PIPE_CAPACITY = 1 * 1024 * 1024  # 1 MB; anything more indicates a broken script

    ctx = mp.get_context('spawn')

    # Create a pipe: parent reads from read_fd, subprocess writes to write_fd.
    result_parent, result_child = ctx.Pipe(duplex=False)
    read_fd = result_parent.fileno()
    write_fd = result_child.fileno()

    sig_r, sig_w = ctx.Pipe(duplex=False)
    signature = secrets.token_hex(16)
    os.write(sig_w.fileno(), signature.encode())
    sig_w.close()

    try:
        import fcntl
        # F_SETPIPE_SZ is Linux-specific (1032); fall back silently on other OSes.
        F_SETPIPE_SZ = 1032
        fcntl.fcntl(write_fd, F_SETPIPE_SZ, _PIPE_CAPACITY)
    except (AttributeError, OSError):
        pass

    parent_tb_conn, child_tb_conn = ctx.Pipe(duplex=False)

    # Make write_fd inheritable before creating the Process so the spawned
    # child receives it as a live, open fd.
    os.set_inheritable(write_fd, True)

    process = ctx.Process(
        target=do_bench_impl,
        args=(
            result_child,
            sig_r,
            qualname,
            test_generator,
            test_args,
            repeats,
            seed,
            None,
            discard,
            nvtx,
            child_tb_conn,
        ),
    )

    process.start()
    child_tb_conn.close()
    result_child.close()

    process.join(timeout=timeout)

    if process.is_alive():
        process.kill()
        process.join()
        result_parent.close()
        raise RuntimeError(
            f"Benchmark subprocess timed out after {timeout}s -- "
            "possible deadlock or infinite loop in kernel"
        )

    if process.exitcode != 0:
        try:
            diagnostic = parent_tb_conn.recv() if parent_tb_conn.poll() else None
        except EOFError:
            diagnostic = None
        parent_tb_conn.close()
        result_parent.close()
        msg = f"Benchmark subprocess failed with exit code {process.exitcode}"
        if diagnostic:
            msg += "\n" + diagnostic
        raise RuntimeError(msg)

    # Child has exited and closed its write-end, so this read is bounded.
    raw = os.read(read_fd, _PIPE_CAPACITY)
    result_parent.close()
    parent_tb_conn.close()

    results = BenchmarkResult(None, [-1] * repeats, None, False)
    has_signature = False
    for line in raw.decode().splitlines():
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[0].isdigit():
            iteration = int(parts[0])
            time_us = float(parts[1])
            results.time_us[iteration] = time_us
        elif parts[0] == "event-overhead":
            results.event_overhead_us = float(parts[1].split()[0])
        elif parts[0] == "error-count":
            results.errors = int(parts[1])
        elif parts[0] == "signature":
            if signature != parts[1]:
                raise AssertionError(f"Invalid signature")
            has_signature = True
    if not has_signature:
        raise RuntimeError(f"No signature found in output")

    results.full = all((t > 0 for t in results.time_us))
    return results
