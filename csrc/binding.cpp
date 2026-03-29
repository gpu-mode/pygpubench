// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <utility>
#include <random>
#include <thread>
#include "manager.h"
#include "utils.h"

int supervisor_main(int sock_fd);

namespace nb = nanobind;


void do_bench(int result_fd, int input_fd, const std::string& kernel_qualname, const nb::object& test_generator,
              const nb::dict& test_kwargs, std::uintptr_t stream, bool discard, bool nvtx, bool landlock, bool mseal,
              int supervisor_sock_fd) {
    std::vector<char> signature_bytes(32);
    auto config = read_benchmark_parameters(input_fd, signature_bytes.data());
    auto mgr = make_benchmark_manager(result_fd, signature_bytes, config.Seed, discard, nvtx, landlock, mseal, supervisor_sock_fd);
    cleanse(signature_bytes.data(), 32);

    {
        nb::gil_scoped_release release;
        std::exception_ptr thread_exception;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        std::thread run_thread ([&]()
        {
            try {
                 CUDA_CHECK(cudaSetDevice(device));
                 nb::gil_scoped_acquire acquire;
                 auto [args, expected] = mgr->setup_benchmark(nb::cast<nb::callable>(test_generator), test_kwargs, config.Repeats);
                 mgr->do_bench_py(kernel_qualname, args, expected, reinterpret_cast<cudaStream_t>(stream));
             } catch (...) {
                 thread_exception = std::current_exception();
             }
        });
        run_thread.join();
        if (thread_exception)
            std::rethrow_exception(thread_exception);
    }

    mgr->send_report();
    mgr->clean_up();
}


NB_MODULE(_pygpubench, m) {
    m.def("do_bench", do_bench,
          nb::arg("result_fd"),
          nb::arg("input_fd"),
          nb::arg("kernel_qualname"),
          nb::arg("test_generator"),
          nb::arg("test_kwargs"),
          nb::arg("stream"),
          nb::arg("discard")            = true,
          nb::arg("nvtx")               = false,
          nb::arg("landlock")           = true,
          nb::arg("mseal")              = true,
          nb::arg("supervisor_sock_fd") = -1   // -1 = seccomp disabled
    );

    m.def("run_supervisor", [](int sock_fd) {
        nb::gil_scoped_release release;
        supervisor_main(sock_fd);
    }, nb::arg("sock_fd"));
}
