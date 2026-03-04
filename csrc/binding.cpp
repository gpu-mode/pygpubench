// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <utility>
#include "manager.h"

namespace nb = nanobind;


void do_bench(int result_fd, int input_fd, const std::string& kernel_qualname, const nb::object& test_generator, const nb::dict& test_kwargs, std::uintptr_t stream, bool discard, bool nvtx) {
    auto config = read_benchmark_parameters(input_fd);
    BenchmarkManager mgr(result_fd, config.Signature, config.Seed, discard, nvtx);
    auto [args, expected] = mgr.setup_benchmark(nb::cast<nb::callable>(test_generator), test_kwargs,  config.Repeats);
    mgr.do_bench_py(kernel_qualname, args, expected, reinterpret_cast<cudaStream_t>(stream));
}


NB_MODULE(_pygpubench, m) {
    m.def("do_bench", do_bench);
}
