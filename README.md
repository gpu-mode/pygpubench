[![PyPI version](https://img.shields.io/pypi/v/pygpubench)](https://pypi.org/project/pygpubench/)

# PyGPUBench

Utilities for benchmarking low-latency CUDA kernels in an _adversarial_ setting.
Contrary to many existing benchmarking tools, which generally assume a cooperative kernel that
can be tested and benchmarked independently, this library tries to defend against kernels that
try to exploit benchmarking flaws to receive higher scores.

## Usage
To benchmark a kernel, two ingredients are needed:
1. The qualified name of the kernel function. It is important that the testing script itself does not import the kernel function, as this implies executing untrusted code.
2. A function that generates test/benchmark inputs. This function takes keyword arguments of configuration parameters,
   as well as the reserved argument `seed` to randomize the problem. It returns two tuples:
   The first contains the inputs for the kernel and will
   be used to call the kernel function, and the second contains the expected output and the required absolute and relative tolerance.

```python
import torch
import pygpubench

def generate_input(**kwargs):
    ...

def reference_kernel(args):
    ...

def generate_test_case(*, seed, **kwargs):
    x, y = generate_input(**kwargs, seed=seed)
    expected = torch.empty_like(y)
    reference_kernel((expected, x))
    return (y, x), (expected, 1e-6, 1e-6)


res = pygpubench.do_bench_isolated("submission.kernel", generate_test_case,  {"size": 1024}, 100, 5, discard=True)
print("❌" if res.errors else "✅", pygpubench.basic_stats(res.time_us))
```
For the full example see [grayscale.py](test/grayscale.py)


## Implementation
Unfortunately, any benchmarking tool written in python is inherently vulnerable to monkeypatching
and `inpect`-based manipulation of its variables by its callees. 
Therefore, PyGPUBench implements its main benchmarking logic in a compiled C++ extension. 
While this still leaves vulnerabilities - the code is running in the same address space, after all –
it makes attacks require much more sophistication. Running in a separate process fundamentally
clashes with the desire to benchmark very short kernels; cuda events must be recorded in the same
process as the kernel. Fortunately, we can assume that a reward-hacking LLM is still rather 
unlikely to produce a compiled extension that runs sophisticated low-level exploits.

Note that, as soon as any user code is executed, the entire python runtime becomes untrustworthy.
Consequently, benchmark results are not returned to python, but instead written to a file. The
name of this file is passed as an argument to the benchmarking function, and the file is unlinked
before the user code is called, making it impossible to reopen this file.
The `do_bench_isolated` function is designed to streamline this process: It automates creating
the temporary file, spawning a new python process to handle benchmarking and reading the
results back into python (the original, untainted process).

Thus, the library provides two main interfaces to benchmarking:
`do_bench_impl` runs benchmarking directly in the current process,
`do_bench_isolated` runs it in a separate process and automaticallly handles 
I/O through a temporary file.

Additional measures to mitigate benchmark cheating are that benchmark inputs are generated before any benchmark is run,
but then moved to a GPU memory location unknown to `torch` (allocated directly with cudaMalloc in C++). Only before
the actual kernel is launched do we copy the inputs back to their original locations. Problematically, this would put the
inputs into L2 cache, which we want to avoid. This means that between the copy and the kernel launch, there has to be another
kernel that clears the L2 cache, opening a window of opportunity for cheating. To minimize the duration of vulnerability,
we put a small fraction of random canaries into the input data, that is, a subset of memory location contains wrong data.
Only after L2 clearing do we fix up these values; this pulls them into L2 cache, but since they make up less than 1% of
the total data, we consider this an acceptable tradeoff.

Similarly, after the kernel is finished, we directly launch the testing kernel with a programmatically-dependent launch,
again to minimize the window of opportunity for cheating by writing results from a different stream. This could have a
small effect on performance, as during the tail of the user kernel blocks of the test kernel are already put on the SMs
and generate memory traffic. In the checking kernel, the order in which blocks are checked is randomized, so that it is
not a viable strategy to only write the later blocks of the result from an unsynchronized stream.
