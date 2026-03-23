// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "manager.h"
#include "utils.h"
#include "check.h"
#include <chrono>
#include <cuda_runtime.h>
#include <optional>
#include <system_error>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <random>
#include <thread>
#include <nvtx3/nvToolsExt.h>
#include <nanobind/stl/string.h>
#include <sys/mman.h>

static constexpr std::size_t ArenaSize = 2 * 1024 * 1024;

extern void clear_cache(void* dummy_memory, int size, bool discard, cudaStream_t stream);
extern void install_landlock();
extern bool mseal_supported();
extern void seal_executable_mappings();
extern void install_seccomp_filter();
extern void seccomp_protect_page_range(uintptr_t protected_page, size_t page_size);

static void check_check_approx_match_dispatch(unsigned* result, void* expected_data, nb::dlpack::dtype expected_type,
                                       const nb_cuda_array& received, float r_tol, float a_tol, unsigned seed, std::size_t n_bytes, cudaStream_t stream) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    nb::dlpack::dtype fp16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    nb::dlpack::dtype fp32_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    if (expected_type == bf16_dt) {
        check_approx_match_launcher(result, static_cast<const nv_bfloat16*>(expected_data), static_cast<const nv_bfloat16*>(received.data()), r_tol, a_tol, seed, n_bytes / 2, stream);
    } else if (expected_type == fp16_dt) {
        check_approx_match_launcher(result, static_cast<const half*>(expected_data), static_cast<const half*>(received.data()), r_tol, a_tol, seed, n_bytes / 2, stream);
    } else if (expected_type == fp32_dt) {
        check_approx_match_launcher(result, static_cast<const float*>(expected_data), static_cast<const float*>(received.data()), r_tol, a_tol, seed, n_bytes / 4, stream);
    } else {
        throw std::runtime_error("Unsupported dtype for check_approx_match");
    }
}

static nb::callable kernel_from_qualname(const std::string& qualname) {
    const auto dot = qualname.rfind('.');
    if (dot == std::string::npos) {
        throw std::invalid_argument(
            "qualname must be a fully qualified name (e.g. 'my_module.kernel'), got: " + qualname
        );
    }
    const std::string module_name = qualname.substr(0, dot);
    const std::string attr = qualname.substr(dot + 1);
    if (module_name.empty() || attr.empty()) {
        throw std::invalid_argument(
            "qualname has empty module or attribute part: " + qualname
        );
    }
    nb::object mod = nb::module_::import_("importlib").attr("import_module")(module_name);
    return nb::cast<nb::callable>(mod.attr(attr.c_str()));
}

static void trigger_gc() {
    // Get the gc module and call collect()
    nb::module_ gc = nb::module_::import_("gc");
    (void)gc.attr("collect")();
}

BenchmarkParameters read_benchmark_parameters(int input_fd, void* signature_out) {
    char buf[256];
    FILE* inp_file = fdopen(input_fd, "r");
    if (!inp_file) {
        throw std::system_error(errno, std::generic_category(), "Could not open input pipe");
    }

    auto read_line = [&](const char* field_name) {
        if (!fgets(buf, sizeof(buf), inp_file)) {
            int err = errno;
            if (feof(inp_file)) {
                fclose(inp_file);
                throw std::runtime_error(std::string("Unexpected EOF reading ") + field_name);
            }
            fclose(inp_file);
            throw std::system_error(err, std::generic_category(),
                std::string("Could not read ") + field_name);
        }
    };

    if (fread(signature_out, 1, 32, inp_file) != 32) {
        if (feof(inp_file)) {
            fclose(inp_file);
            throw std::runtime_error("Unexpected EOF reading signature (got fewer than 32 bytes)");
        }
        fclose(inp_file);
        throw std::system_error(errno, std::generic_category(), "fread failed reading signature");
    }

    if (fgetc(inp_file) != '\n') {
        fclose(inp_file);
        throw std::runtime_error("Expected newline after signature");
    }

    read_line("seed");
    char* end;
    std::uint64_t seed = std::strtoull(buf, &end, 10);
    if (end == buf || (*end != '\n' && *end != '\0')) {
        fclose(inp_file);
        throw std::invalid_argument("Invalid seed: " + std::string(buf));
    }

    read_line("repeats");
    long repeats = std::strtol(buf, nullptr, 10);
    if (repeats >= std::numeric_limits<int>::max() || repeats < 2) {
        fclose(inp_file);
        throw std::invalid_argument(
            "Invalid number of repeats: " + std::to_string(repeats));
    }

    fclose(inp_file);
    return {seed, static_cast<int>(repeats)};
}

BenchmarkManager::BenchmarkManager(int result_fd, ObfuscatedHexDigest signature, std::uint64_t seed, bool discard,
                                   bool nvtx, bool landlock, bool mseal, int supervisor_socket) : mSignature(std::move(signature)) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&mL2CacheSize, cudaDevAttrL2CacheSize, device));
    CUDA_CHECK(cudaMalloc(&mDeviceDummyMemory, 2 * mL2CacheSize));
    // allocate a large arena (2MiB) to place the error counter in
    CUDA_CHECK(cudaMalloc(&mDeviceErrorBase, ArenaSize));
    mOutputPipe = fdopen(result_fd, "w");
    if (!mOutputPipe) {
        throw std::runtime_error("Could not open output pipe");
    }

    mNVTXEnabled = nvtx;
    mLandlock = landlock;
    mSeal = mseal;
    mDiscardCache = discard;
    mSeed = seed;
}

BenchmarkManager::~BenchmarkManager() {
    if (mOutputPipe) {
        fclose(mOutputPipe);
        mOutputPipe = nullptr;
    }
    cudaFree(mDeviceDummyMemory);
    cudaFree(mDeviceErrorBase);
    for (auto& event : mStartEvents) cudaEventDestroy(event);
    for (auto& event : mEndEvents) cudaEventDestroy(event);
    for (auto& exp: mExpectedOutputs) cudaFree(exp.Value);
}

std::pair<std::vector<nb::tuple>, std::vector<nb::tuple>> BenchmarkManager::setup_benchmark(const nb::callable& generate_test_case, const nb::dict& kwargs, int repeats) {
    std::mt19937_64 rng(mSeed);
    std::uniform_int_distribution<std::uint64_t> dist(0, std::numeric_limits<std::uint64_t>::max());
    // generate one more input to handle warmup
    std::vector<nb::tuple> kernel_args(repeats + 1);
    std::vector<nb::tuple> expected(repeats + 1);
    for (int i = 0; i < repeats + 1; i++) {
        // create new copy of the kwargs dict
        nb::dict call_kwargs;
        for (auto [k, v] : kwargs) {
            // Disallow user-specified "seed" to avoid silently overwriting it below.
            if (nb::cast<std::string>(k) == "seed") {
                throw std::runtime_error("The 'seed' keyword argument is reserved and must not be passed in kwargs.");
            }
            call_kwargs[k] = v;
        }
        call_kwargs["seed"] = dist(rng);

        auto gen = nb::cast<nb::tuple>(generate_test_case(**call_kwargs));
        kernel_args[i] = nb::cast<nb::tuple>(gen[0]);
        expected[i] = nb::cast<nb::tuple>(gen[1]);
    }
    return std::make_pair(std::move(kernel_args), std::move(expected));
}

bool can_convert_to_tensor(nb::handle obj) {
    return nb::isinstance<nb_cuda_array>(obj);
}

auto BenchmarkManager::make_shadow_args(const nb::tuple& args, cudaStream_t stream) -> std::vector<std::optional<ShadowArgument>> {
    std::vector<std::optional<ShadowArgument>> shadow_args(args.size());
    int nargs = args.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned> canary_seed_dist(0,  0xffffffff);
    for (int i = 1; i < nargs; i++) {
        if (can_convert_to_tensor(args[i])) {
            nb_cuda_array arr = nb::cast<nb_cuda_array>(args[i]);
            void* shadow;
            CUDA_CHECK(cudaMalloc(&shadow, arr.nbytes()));
            CUDA_CHECK(cudaMemcpyAsync(shadow, arr.data(), arr.nbytes(), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(arr.data(), 0xff, arr.nbytes(), stream));
            unsigned seed = canary_seed_dist(gen);
            shadow_args[i] = ShadowArgument{nb::cast<nb_cuda_array>(args[i]), shadow, seed};
            canaries(shadow, arr.nbytes(), seed, stream);
        }
    }
    return shadow_args;
}

void BenchmarkManager::nvtx_push(const char* name) {
    if (mNVTXEnabled)
        nvtxRangePush(name);
}

void BenchmarkManager::nvtx_pop() {
    if (mNVTXEnabled)
        nvtxRangePop();
}

void BenchmarkManager::validate_result(Expected& expected, const nb_cuda_array& result, unsigned seed, cudaStream_t stream) {
    if (expected.Mode == Expected::ExactMatch) {
        check_exact_match_launcher(
            mDeviceErrorCounter,
            static_cast<std::byte*>(expected.Value),
            static_cast<std::byte*>(result.data()),
            seed,
            expected.Size, stream);
    } else {
        check_check_approx_match_dispatch(
            mDeviceErrorCounter,
            expected.Value, expected.DType, result,
            expected.RTol, expected.ATol, seed, expected.Size, stream);
    }
}

void BenchmarkManager::clear_cache(cudaStream_t stream) {
    ::clear_cache(mDeviceDummyMemory, 2 * mL2CacheSize, mDiscardCache, stream);
}

BenchmarkManager::ShadowArgument::ShadowArgument(nb_cuda_array original, void* shadow, unsigned seed) :
    Original(std::move(original)), Shadow(shadow), Seed(seed) {
}

BenchmarkManager::ShadowArgument::~ShadowArgument() {
    if (Shadow != nullptr)
        cudaFree(Shadow);
}

BenchmarkManager::ShadowArgument::ShadowArgument(ShadowArgument&& other) noexcept :
    Original(std::move(other.Original)), Shadow(std::exchange(other.Shadow, nullptr)), Seed(other.Seed) {
}

BenchmarkManager::ShadowArgument& BenchmarkManager::ShadowArgument::operator=(ShadowArgument&& other) noexcept {
    Original = std::move(other.Original);
    Shadow = std::exchange(other.Shadow, nullptr);
    Seed = other.Seed;
    return *this;
}

void BenchmarkManager::install_protections() {
    // clean up as much python state as we can
    trigger_gc();

    // restrict access to file system
    if (mLandlock)
        install_landlock();

    if (mSeal) {
        if (!mseal_supported()) {
            throw std::runtime_error("mseal=True but kernel does not support sealing executable mappings");
        }
        seal_executable_mappings();
    }

    install_seccomp_filter();
}

int BenchmarkManager::run_warmup(nb::callable& kernel, const nb::tuple& args, cudaStream_t stream) {
    std::chrono::high_resolution_clock::time_point cpu_start = std::chrono::high_resolution_clock::now();
    int warmup_run_count = 0;
    double time_estimate;
    nvtx_push("timing");
    while (true) {
        // note: we are assuming here that calling the kernel multiple times for the same input is a safe operation
        // this is only potentially problematic for in-place kernels;
        CUDA_CHECK(cudaDeviceSynchronize());
        clear_cache(stream);
        kernel(*args);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = cpu_end - cpu_start;
        ++warmup_run_count;
        if (elapsed_seconds.count() > mWarmupSeconds) {
            time_estimate = elapsed_seconds.count() / warmup_run_count;
            break;
        }
    }
    nvtx_pop();

    // note: this is a very conservative estimate. Timing above was measured with syncs between every kernel.
    int calls = mOutputBuffers.size() - 1;
    const int actual_calls = std::clamp(static_cast<int>(std::ceil(mBenchmarkSeconds / time_estimate)), 1, calls);

    if (actual_calls < 3) {
        throw std::runtime_error("The initial speed test indicated that running times are too slow to generate meaningful benchmark numbers: " + std::to_string(time_estimate));
    }

    return actual_calls;
}

void protect_range(void* ptr, size_t size, int prot) {
    std::uintptr_t start = reinterpret_cast<std::uintptr_t>(ptr) & ~4095;
    std::uintptr_t end   = (reinterpret_cast<std::uintptr_t>(ptr) + size + 4095) & ~4095;
    if (mprotect(reinterpret_cast<void*>(start), end - start, prot) < 0)
        throw std::system_error(errno, std::system_category(), "mprotect");
}

nb::callable BenchmarkManager::get_kernel(const std::string& qualname) {
    nb::gil_scoped_release release;
    std::uintptr_t lo = reinterpret_cast<std::uintptr_t>(this) & ~4095;
    std::uintptr_t hi = (reinterpret_cast<std::uintptr_t>(this) + sizeof(BenchmarkManager) + 4095) & ~4095;

    nb::callable kernel;
    std::exception_ptr thread_exception;
    // make the BenchmarkManager inaccessible
    protect_range(reinterpret_cast<void*>(lo), hi - lo, PROT_NONE);
    // TODO make stack inaccessible (may be impossible) or read-only during the call
    // call the python kernel generation function from a different thread.

    std::thread make_kernel_thread([&]() {
        try {
            // new thread, new seccomp.
            seccomp_protect_page_range(lo, hi - lo);
            nb::gil_scoped_acquire guard;
            kernel = kernel_from_qualname(qualname);
        } catch (...) {
                thread_exception = std::current_exception();
        }
    });

    make_kernel_thread.join();
    // make it accessible again. This is in the original thread, so the tightened seccomp
    // policy does not apply here.
    protect_range(reinterpret_cast<void*>(lo), hi - lo, PROT_READ | PROT_WRITE);

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    return kernel;
}

void BenchmarkManager::do_bench_py(
        const std::string& kernel_qualname,
        const std::vector<nb::tuple>& args,
        const std::vector<nb::tuple>& expected,
        cudaStream_t stream)
{
    setup_test_cases(args, expected, stream);
    install_protections();

    // at this point, we call user code as we import the kernel (executing arbitrary top-level code)
    // after this, we cannot trust python anymore
    nb::callable kernel = get_kernel(kernel_qualname);

    // ok, first run for compilations etc
    nvtx_push("warmup");
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel(*args.at(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtx_pop();

    // now, run a few more times for warmup; in total aim for 1 second of warmup runs
    int actual_calls = run_warmup(kernel, args.at(0), stream);
    constexpr int DRY_EVENTS = 100;
    const int num_events = std::max(actual_calls, DRY_EVENTS);
    mStartEvents.resize(num_events);
    mEndEvents.resize(num_events);
    for (int i = 0; i < num_events; i++) {
        CUDA_CHECK(cudaEventCreate(&mStartEvents.at(i)));
        CUDA_CHECK(cudaEventCreate(&mEndEvents.at(i)));
    }

    // pick a random spot for the unsigned
    // initialize the whole area with random junk; the error counter
    // will be shifted by the initial value, so just writing zero
    // won't result in passing the tests.
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::ptrdiff_t> dist(0, ArenaSize / sizeof(unsigned) - 1);
    std::uniform_int_distribution<unsigned> noise_generator(0, std::numeric_limits<unsigned>::max());
    std::vector<unsigned> noise(ArenaSize / sizeof(unsigned));
    std::generate(noise.begin(), noise.end(), [&]() -> unsigned { return noise_generator(rng); });
    CUDA_CHECK(cudaMemcpyAsync(mDeviceErrorBase, noise.data(), noise.size() * sizeof(unsigned), cudaMemcpyHostToDevice,  stream));
    std::ptrdiff_t offset = dist(rng);
    mDeviceErrorCounter = mDeviceErrorBase + offset;
    mErrorCountShift = noise.at(offset);

    // dry run -- measure overhead of events
    float median_event_time = measure_event_overhead(DRY_EVENTS, stream);
    fprintf(mOutputPipe, "event-overhead\t%f µs\n", median_event_time * 1000);

    // create a randomized order for running the tests
    std::vector<int> test_order(actual_calls);
    std::iota(test_order.begin(), test_order.end(), 1);
    std::shuffle(test_order.begin(), test_order.end(), rng);

    std::uniform_int_distribution<unsigned> check_seed_generator(0,  0xffffffff);

    nvtx_push("benchmark");
    // now do the real runs
    for (int i = 0; i < actual_calls; i++) {
        int test_id = test_order.at(i);
        // page-in real inputs. If the user kernel runs on the wrong stream, it's likely it won't see the correct inputs
        // unfortunately, we need to do this before clearing the cache, so there is a window of opportunity
        // *but* we deliberately modify a small subset of the inputs, which only get corrected immediately before
        // the user code call.
        for (auto& shadow_arg : mShadowArguments.at(test_id)) {
            if (shadow_arg) {
                CUDA_CHECK(cudaMemcpyAsync(shadow_arg->Original.data(), shadow_arg->Shadow, shadow_arg->Original.nbytes(), cudaMemcpyDeviceToDevice, stream));
            }
        }

        nvtx_push("cc");
        clear_cache(stream);
        nvtx_pop();

        // ok, now we revert the canaries. This _does_ bring in the corresponding cache lines,
        // but they are very sparse (1/256), so that seems like an acceptable trade-off
        for (auto& shadow_arg : mShadowArguments.at(test_id)) {
            if (shadow_arg) {
                canaries(shadow_arg->Original.data(), shadow_arg->Original.nbytes(), shadow_arg->Seed, stream);
            }
        }

        CUDA_CHECK(cudaEventRecord(mStartEvents.at(i), stream));
        nvtx_push("kernel");
        (void)kernel(*args.at(test_id));
        nvtx_pop();
        CUDA_CHECK(cudaEventRecord(mEndEvents.at(i), stream));
        // immediately after the kernel, launch the checking code; if there is some unsynced work done on another stream,
        // this increases the chance of detection.
        validate_result(mExpectedOutputs.at(test_id), mOutputBuffers.at(test_id), check_seed_generator(rng), stream);
    }
    nvtx_pop();

    cudaEventSynchronize(mEndEvents.back());
    unsigned error_count;
    CUDA_CHECK(cudaMemcpy(&error_count, mDeviceErrorCounter, sizeof(unsigned), cudaMemcpyDeviceToHost));
    // subtract the nuisance shift that we applied to the counter
    error_count -= mErrorCountShift;

    #ifdef ENABLE_EXPLOIT_TARGET
    if (g_exploit_target != 0xDEADBEEFCAFEBABE) {
        fprintf(mOutputPipe, "error-count\t%u\n", 0);
        for (int i = 0; i < actual_calls; i++) {
            fprintf(mOutputPipe, "%d\t%f\n", test_order.at(i) - 1, 10.f);
        }
    } else {
        fprintf(mOutputPipe, "error-count\t%u\n", 42424242);
    }
    #else
    if (error_count > 0) {
        fprintf(mOutputPipe, "error-count\t%u\n", error_count);
    }

    for (int i = 0; i < actual_calls; i++) {
        float duration;
        CUDA_CHECK(cudaEventElapsedTime(&duration, mStartEvents.at(i), mEndEvents.at(i)));
        fprintf(mOutputPipe, "%d\t%f\n", test_order.at(i) - 1, duration * 1000);
    }
    #endif

    fprintf(mOutputPipe, "signature\t");
    fwrite(mSignature.data(), 1, 32, mOutputPipe);
    fputc('\n', mOutputPipe);
    fflush(mOutputPipe);

    // cleanup events
    for (auto& event : mStartEvents) CUDA_CHECK(cudaEventDestroy(event));
    for (auto& event : mEndEvents) CUDA_CHECK(cudaEventDestroy(event));
    mStartEvents.clear();
    mEndEvents.clear();
}

void BenchmarkManager::setup_test_cases( const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected, cudaStream_t stream) {
    if (args.size() < 5) {
        throw std::runtime_error("Not enough test cases to run benchmark");
    }
    if (expected.size() != args.size()) {
        throw std::runtime_error("Expected results and test case list do not have the same length");
    }

    // extract relevant infos from args and expected
    // by convention, the first arg is the output tensor.
    // TODO handle multiple outputs
    mOutputBuffers.resize(args.size());
    for (int i = 0; i < args.size(); i++) {
        mOutputBuffers.at(i) = nb::cast<nb_cuda_array>(args.at(i)[0]);
    }

    // Generate "shadow" copies of input arguments
    for (const auto & arg : args) {
        mShadowArguments.emplace_back(make_shadow_args(arg, stream));
    }

    // prepare expected outputs
    setup_expected_outputs(args, expected);
}

float BenchmarkManager::measure_event_overhead(int repeats, cudaStream_t stream) {
    nvtx_push("dry-run");
    // ensure that the GPU is busy for a short moment, so we can submit all the events
    // before the GPU reaches them
    clear_cache(stream);
    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventRecord(mStartEvents.at(i), stream));
        CUDA_CHECK(cudaEventRecord(mEndEvents.at(i), stream));
    }
    nvtx_pop();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> empty_event_times(repeats);
    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventElapsedTime(empty_event_times.data() + i, mStartEvents.at(i), mEndEvents.at(i)));
    }
    std::sort(empty_event_times.begin(), empty_event_times.end());
    float median = empty_event_times.at(empty_event_times.size() / 2);
    return median;
}

void BenchmarkManager::setup_expected_outputs(const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected) {
    mExpectedOutputs.resize(args.size());
    for (int i = 0; i < args.size(); i++) {
        const nb::tuple& expected_tuple = expected.at(i);
        nb_cuda_array expected_array = nb::cast<nb_cuda_array>(expected_tuple[0]);

        // make a copy of the expected result and put it in memory not owned by torch; overwrite the original
        // so it cannot be read by cheating solutions.
        void* copy_mem;
        CUDA_CHECK(cudaMalloc(&copy_mem, expected_array.nbytes()));
        CUDA_CHECK(cudaMemcpy(copy_mem, expected_array.data(), expected_array.nbytes(), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(expected_array.data(), 0, expected_array.nbytes()));

        if (expected.at(i).size() == 1) {
            mExpectedOutputs.at(i) = {Expected::ExactMatch, copy_mem, expected_array.nbytes(), expected_array.dtype(), 0.f, 0.f};
        } else {
            float rtol = nb::cast<float>(expected_tuple[1]);
            float atol = nb::cast<float>(expected_tuple[2]);
            mExpectedOutputs.at(i) = {Expected::ApproxMatch, copy_mem, expected_array.nbytes(), expected_array.dtype(), atol, rtol};
        }
    }
}
