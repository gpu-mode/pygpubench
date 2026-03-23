// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PYGPUBENCH_SRC_MANAGER_H
#define PYGPUBENCH_SRC_MANAGER_H

#include <functional>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>
#include <optional>
#include <nanobind/nanobind.h>
#include "nanobind/ndarray.h"

#include "obfuscate.h"

namespace nb = nanobind;

using nb_cuda_array = nb::ndarray<nb::c_contig, nb::device::cuda>;

struct BenchmarkParameters {
    std::uint64_t Seed;
    int Repeats;
};

BenchmarkParameters read_benchmark_parameters(int input_fd, void* signature_out);

class BenchmarkManager {
public:
    BenchmarkManager(int result_fd, ObfuscatedHexDigest signature, std::uint64_t seed, bool discard, bool nvtx, bool landlock, bool mseal, int supervisor_socket);
    ~BenchmarkManager();
    std::pair<std::vector<nb::tuple>, std::vector<nb::tuple>> setup_benchmark(const nb::callable& generate_test_case, const nb::dict& kwargs, int repeats);
    void do_bench_py(const std::string& kernel_qualname, const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected, cudaStream_t stream);
private:
    struct Expected {
        enum EMode {
            ExactMatch,
            ApproxMatch
        } Mode;
        void* Value = nullptr;
        std::size_t Size;
        nb::dlpack::dtype DType;
        float ATol;
        float RTol;
    };

    struct ShadowArgument {
        nb_cuda_array Original;
        void* Shadow = nullptr;
        unsigned Seed = -1;
        ShadowArgument(nb_cuda_array original, void* shadow, unsigned seed);
        ~ShadowArgument();
        ShadowArgument(ShadowArgument&& other) noexcept;
        ShadowArgument& operator=(ShadowArgument&& other) noexcept;
    };

    using ShadowArgumentList = std::vector<std::optional<ShadowArgument>>;

    double mWarmupSeconds = 1.0;
    double mBenchmarkSeconds = 1.0;

    std::vector<cudaEvent_t> mStartEvents;
    std::vector<cudaEvent_t> mEndEvents;

    std::chrono::high_resolution_clock::time_point mCPUStart;

    int* mDeviceDummyMemory = nullptr;
    int mL2CacheSize;
    unsigned* mDeviceErrorCounter = nullptr;
    unsigned* mDeviceErrorBase = nullptr;
    unsigned mErrorCountShift = 0;
    bool mNVTXEnabled = false;
    bool mDiscardCache = true;
    bool mLandlock = true;
    bool mSeal = true;
    int mSupervisorSock = -1;
    std::uint64_t mSeed = -1;
    std::vector<Expected> mExpectedOutputs;
    std::vector<ShadowArgumentList> mShadowArguments;
    std::vector<nb_cuda_array> mOutputBuffers;

    FILE* mOutputPipe = nullptr;
    ObfuscatedHexDigest mSignature;

    static ShadowArgumentList make_shadow_args(const nb::tuple& args, cudaStream_t stream);

    void nvtx_push(const char* name);
    void nvtx_pop();

    void validate_result(Expected& expected, const nb_cuda_array& result, unsigned seed, cudaStream_t stream);
    void clear_cache(cudaStream_t stream);
    float measure_event_overhead(int repeats, cudaStream_t stream);
    void setup_expected_outputs(const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected);
    void setup_test_cases(const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected, cudaStream_t stream);

    void install_protections();
    int run_warmup(nb::callable& kernel, const nb::tuple& args, cudaStream_t stream);
    nb::callable get_kernel(const std::string& qualname);


    // debug only: Any sort of test exploit that targets specific values of this class is going to be brittle,
    // because simple refactoring will break the exploit, even though it does not close the underlying vulnerability.
    // so instead, we use this canary value: If an exploit is able to manipulate this value, it is probably also
    // able to do a real cheat. But we can test this much easier.
#ifdef ENABLE_EXPLOIT_TARGET
    // Known canary value for exploit testing
    volatile uint64_t g_exploit_target = 0xDEADBEEFCAFEBABE;
#endif
};

#endif //PYGPUBENCH_SRC_MANAGER_H
