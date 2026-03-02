import dataclasses
import math
import multiprocessing as mp
import tempfile
import traceback

from pathlib import Path
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
    "KernelGeneratorInterface",
    "TestGeneratorInterface",
    "ExpectedResult",
]


def do_bench_impl(out_file: str, kernel_generator: KernelGeneratorInterface, test_generator: TestGeneratorInterface,
                  test_args: dict, repeats: int, seed: int, stream: int = None, discard: bool = True,
                  unlink: bool = False, nvtx: bool = False, tb_conn=None):
    """
    Benchmarks the kernel returned by `kernel_generator` against the test case returned by `test_generator`.
    :param out_file: File in which to write the benchmark results.
    :param kernel_generator: A function that takes no arguments and returns a kernel function.
    :param test_generator: A function that takes the test arguments (including a seed) and returns a test case; i.e., a tuple of (input, expected)
    :param test_args: keyword arguments to be passed to `test_generator`. Seed will be generated automatically.
    :param repeats: Number of times to repeat the benchmark. `test_generator` will be called `repeats` times.
    :param stream: Cuda stream on which to run the benchmark. If not given, torch's current stream is selected
    :param discard: If true, then cache lines are discarded as part of cache clearing before each benchmark run.
    :param unlink: Whether to unlink the output file before calling `kernel_generator`. Unlinking makes it impossible to
    open the file again, protecting it against malicious kernels.
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
                out_file,
                kernel_generator,
                test_generator,
                test_args,
                repeats,
                seed,
                stream,
                discard,
                unlink,
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
        kernel_generator: KernelGeneratorInterface,
        test_generator: TestGeneratorInterface,
        test_args: dict,
        repeats: int,
        seed: int,
        *,
        discard: bool = True,
        nvtx: bool = False
) -> BenchmarkResult:
    """
    Runs kernel benchmark (`do_bench_impl`) in a subprocess for proper isolation.
    """
    assert repeats > 1

    # Create a named temporary file for the C++ extension to store the results in
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
        result_file = f.name

    try:
        # open file before running subprocess; process will unlink
        with open(result_file, 'r') as f:
            # Spawn testing process
            ctx = mp.get_context('spawn')
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=do_bench_impl,
                args=(
                    result_file,
                    kernel_generator,
                    test_generator,
                    test_args,
                    repeats,
                    seed,
                    None,
                    discard,
                    True,   # unlink=True
                    nvtx,
                    child_conn,
                ),
            )

            process.start()
            child_conn.close()
            process.join()

            if process.exitcode != 0:
                diagnostic = parent_conn.recv() if parent_conn.poll() else None
                parent_conn.close()
                msg = f"Benchmark subprocess failed with exit code {process.exitcode}"
                if diagnostic:
                    msg += "\n" + diagnostic
                raise RuntimeError(msg)

            # Read results from file
            results = BenchmarkResult(None, [-1] * repeats, None, False)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2 and parts[0].isdigit():
                    iteration = int(parts[0])
                    time_us = float(parts[1])
                    results.time_us[iteration] = time_us
                elif parts[0] == "event-overhead":
                    results.event_overhead_us = float(parts[1].split()[0])
                elif parts[0] == "error-count":
                    results.errors = int(parts[1])
            parent_conn.close()
            results.full = all((t > 0 for t in results.time_us))
        return results

    finally:
        Path(result_file).unlink(missing_ok=True)
