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
#include <sstream>
#include <thread>
#include <nvtx3/nvToolsExt.h>
#include <nanobind/stl/string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "protect.h"

static constexpr std::size_t ArenaSize = 2 * 1024 * 1024;

/// Size of the memory used for the benchmark manager's memory (both direct and indirect)
static constexpr std::size_t BenchmarkManagerArenaSize = 128 * 1024 * 1024;

extern void clear_cache(void* dummy_memory, int size, bool discard, cudaStream_t stream);
extern void install_landlock();
extern bool mseal_supported();
extern void seal_mappings();
extern bool supports_seccomp_notify();
extern void install_seccomp_filter();
extern void seccomp_install_memory_notify(int supervisor_sock, uintptr_t lo, uintptr_t hi);

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

void BenchmarkManagerDeleter::operator()(BenchmarkManager* p) const noexcept {
    p->~BenchmarkManager();
    if (munmap(static_cast<void*>(p), this->ArenaSize) != 0) {
        std::perror("munmap failed in BenchmarkManagerDeleter");
        std::terminate();
    }
}


BenchmarkManagerPtr make_benchmark_manager(
    int result_fd, const std::vector<char>& signature, std::uint64_t seed,
    bool discard, bool nvtx, bool landlock, bool mseal, int supervisor_socket)
{
    const std::size_t page_size = static_cast<std::size_t>(getpagesize());
    const std::size_t alloc_size = (BenchmarkManagerArenaSize + page_size - 1) & ~(page_size - 1);

    void* mem = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "mmap failed for BenchmarkManager");
    }

    BenchmarkManager* raw = nullptr;
    try {
        raw = new (mem) BenchmarkManager(
            static_cast<std::byte*>(mem), alloc_size,
            result_fd, signature, seed,
            discard, nvtx, landlock, mseal, supervisor_socket);
    } catch (...) {
        // If construction throws, release the mmap'd region before propagating.
        if (munmap(mem, alloc_size) != 0) {
            std::perror("munmap failed in BenchmarkManager");
        }
        throw;
    }
    return {raw, BenchmarkManagerDeleter{alloc_size}};
}



BenchmarkManager::BenchmarkManager(std::byte* arena, std::size_t arena_size,
                                   int result_fd, const std::vector<char>& signature, std::uint64_t seed, bool discard,
                                   bool nvtx, bool landlock, bool mseal, int supervisor_socket)
    : mArena(arena),
      mResource(arena + sizeof(BenchmarkManager),
          arena_size - sizeof(BenchmarkManager),
          std::pmr::null_memory_resource()),

      mSignature(&mResource),
      mSupervisorSock(supervisor_socket),
      mStartEvents(&mResource),
      mEndEvents(&mResource),
      mExpectedOutputs(&mResource),
      mShadowArguments(&mResource),
      mOutputBuffers(&mResource),
      mTestOrder(&mResource)
{
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
    std::random_device rd;
    std::mt19937 rng(rd());
    mSignature.allocate(32, rng);
    std::copy(signature.begin(), signature.end(), mSignature.data());
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

auto BenchmarkManager::make_shadow_args(const nb::tuple& args, cudaStream_t stream,
                                        std::pmr::memory_resource* resource) -> ShadowArgumentList {
    ShadowArgumentList shadow_args(args.size(), resource);
    int nargs = args.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned> canary_seed_dist(0, 0xffffffff);
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
            throw std::runtime_error("mseal=True but kernel does not support sealing memory mappings");
        }
        if (!supports_seccomp_notify()) {
            throw std::runtime_error("mseal=True but kernel does not support seccomp notify, so we cannot enforce memory protection");
        }
        seal_mappings();
    }

    install_seccomp_filter();
}

static void setup_seccomp(int sock, bool install_notify, std::uintptr_t lo, std::uintptr_t hi) {
    if (sock < 0)
        return;
    try {
        if (install_notify)
            seccomp_install_memory_notify(sock, lo, hi);
    } catch (...) {
        close(sock);
        throw;
    }
    close(sock);
}

static double run_warmup_loop(nb::callable& kernel, const nb::tuple& args, cudaStream_t stream,
                              void* cc_memory, std::size_t l2_clear_size, bool discard_cache,
                              double warmup_seconds) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int run_count = 0;

    while (true) {
        ::clear_cache(cc_memory, 2 * l2_clear_size, discard_cache, stream);
        kernel(*args);
        CUDA_CHECK(cudaDeviceSynchronize());

        ++run_count;
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - cpu_start).count();
        if (elapsed > warmup_seconds)
            return elapsed / run_count;
    }
}

nb::callable BenchmarkManager::initial_kernel_setup(double& time_estimate, const std::string& qualname,
                                                     const nb::tuple& call_args, cudaStream_t stream) {
    const std::uintptr_t lo = reinterpret_cast<std::uintptr_t>(mArena);
    const std::uintptr_t hi = lo + BenchmarkManagerArenaSize;

    // snapshot all member state needed in the thread before protecting the arena
    const int sock = mSupervisorSock;
    const bool install_notify = mSeal || supports_seccomp_notify();
    const double warmup_seconds = mWarmupSeconds;
    void* const cc_memory = mDeviceDummyMemory;
    const std::size_t l2_clear_size = mL2CacheSize;
    const bool discard_cache = mDiscardCache;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    nb::callable kernel;
    std::exception_ptr thread_exception;

    nvtx_push("trigger-compile");
    PROTECT_RANGE(lo, hi-lo, PROT_NONE);

    {
        nb::gil_scoped_release release;
        std::thread worker([&] {
            try {
                CUDA_CHECK(cudaSetDevice(device));
                setup_seccomp(sock, install_notify, lo, hi);

                nb::gil_scoped_acquire guard;

                kernel = kernel_from_qualname(qualname);
                CUDA_CHECK(cudaDeviceSynchronize());
                kernel(*call_args);  // trigger JIT compile

                time_estimate = run_warmup_loop(kernel, call_args, stream,
                                                cc_memory, l2_clear_size, discard_cache,
                                                warmup_seconds);
            } catch (...) {
                thread_exception = std::current_exception();
            }
        });
        worker.join();
    }

    PROTECT_RANGE(lo, hi - lo, PROT_READ | PROT_WRITE);
    mSupervisorSock = -1;
    nvtx_pop();

    if (thread_exception)
        std::rethrow_exception(thread_exception);

    return kernel;
}

void BenchmarkManager::randomize_before_test(int num_calls, std::mt19937& rng, cudaStream_t stream) {
    // pick a random spot for the unsigned
    // initialize the whole area with random junk; the error counter
    // will be shifted by the initial value, so just writing zero
    // won't result in passing the tests.
    std::uniform_int_distribution<std::ptrdiff_t> dist(0, ArenaSize / sizeof(unsigned) - 1);
    std::uniform_int_distribution<unsigned> noise_generator(0, std::numeric_limits<unsigned>::max());
    std::vector<unsigned> noise(ArenaSize / sizeof(unsigned));
    std::generate(noise.begin(), noise.end(), [&]() -> unsigned { return noise_generator(rng); });
    CUDA_CHECK(cudaMemcpyAsync(mDeviceErrorBase, noise.data(), noise.size() * sizeof(unsigned), cudaMemcpyHostToDevice,  stream));
    std::ptrdiff_t offset = dist(rng);
    mDeviceErrorCounter = mDeviceErrorBase + offset;
    mErrorCountShift = noise.at(offset);

    // create a randomized order for running the tests
    mTestOrder.resize(num_calls);
    std::iota(mTestOrder.begin(), mTestOrder.end(), 1);
    std::shuffle(mTestOrder.begin(), mTestOrder.end(), rng);
}

void BenchmarkManager::do_bench_py(
        const std::string& kernel_qualname,
        const std::vector<nb::tuple>& args,
        const std::vector<nb::tuple>& expected,
        cudaStream_t stream)
{
    setup_test_cases(args, expected, stream);
    install_protections();

    constexpr std::size_t DRY_EVENTS = 100;
    const std::size_t num_events = std::max(mShadowArguments.size(), DRY_EVENTS);
    mStartEvents.resize(num_events);
    mEndEvents.resize(num_events);
    for (int i = 0; i < num_events; i++) {
        CUDA_CHECK(cudaEventCreate(&mStartEvents.at(i)));
        CUDA_CHECK(cudaEventCreate(&mEndEvents.at(i)));
    }

    // dry run -- measure overhead of events
    mMedianEventTime = measure_event_overhead(DRY_EVENTS, stream);

    double time_estimate = 0.0;
    // at this point, we call user code as we import the kernel (executing arbitrary top-level code)
    // after this, we cannot trust python anymore
    nb::callable kernel = initial_kernel_setup(time_estimate, kernel_qualname, args.at(0), stream);

    int calls = mOutputBuffers.size() - 1;
    const int actual_calls = std::clamp(
        static_cast<int>(std::ceil(mBenchmarkSeconds / time_estimate)), 1, calls);

    if (actual_calls < 3) {
        throw std::runtime_error(
            "The initial speed test indicated that running times are too slow to generate "
            "meaningful benchmark numbers: " + std::to_string(time_estimate));
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    randomize_before_test(actual_calls, rng, stream);
    // from this point on, even the benchmark thread won't write to the arena anymore
    PROTECT_RANGE(mArena, BenchmarkManagerArenaSize, PROT_READ);
    PROTECT_RANGE(mSignature.data(), 4096, PROT_READ);  // make the key fully inaccessible

    std::uniform_int_distribution<unsigned> check_seed_generator(0,  0xffffffff);

    nvtx_push("benchmark");
    // now do the real runs
    for (int i = 0; i < actual_calls; i++) {
        const int test_id = mTestOrder.at(i);
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
}

void BenchmarkManager::send_report() {
    CUDA_CHECK(cudaEventSynchronize(mEndEvents.at(mTestOrder.size() - 1)));
    unsigned error_count;
    CUDA_CHECK(cudaMemcpy(&error_count, mDeviceErrorCounter, sizeof(unsigned), cudaMemcpyDeviceToHost));
    // subtract the nuisance shift that we applied to the counter
    error_count -= mErrorCountShift;

    std::string message = build_result_message(mTestOrder, error_count, mMedianEventTime);
    PROTECT_RANGE(mSignature.data(), 4096, PROT_READ);
    message = encrypt_message(mSignature.data(), 32, message);
    PROTECT_RANGE(mSignature.data(), 4096, PROT_NONE);
    fwrite(message.data(), 1, message.size(), mOutputPipe);
    fflush(mOutputPipe);
}

void BenchmarkManager::clean_up() {
    PROTECT_RANGE(mArena, BenchmarkManagerArenaSize, PROT_READ | PROT_WRITE);

    for (auto& event : mStartEvents) CUDA_CHECK(cudaEventDestroy(event));
    for (auto& event : mEndEvents) CUDA_CHECK(cudaEventDestroy(event));
    mStartEvents.clear();
    mEndEvents.clear();
}

std::string BenchmarkManager::build_result_message(const std::pmr::vector<int>& test_order, unsigned error_count, float median_event_time) const {
    std::ostringstream oss;

    oss << "event-overhead\t" << median_event_time * 1000 << " µs\n";

#ifdef ENABLE_EXPLOIT_TARGET
    if (mExploitCanary != 0xDEADBEEFCAFEBABE) {
        oss << "error-count\t0\n";
        for (int i : test_order) {
            oss << (i - 1) << "\t10.000000\n";
        }
    } else {
        oss << "error-count\t42424242\n";
    }
#else
    if (error_count > 0) {
        oss << "error-count\t" << error_count << "\n";
    }
    for (int i = 0; i < test_order.size(); i++) {
        float duration;
        CUDA_CHECK(cudaEventElapsedTime(&duration, mStartEvents.at(i), mEndEvents.at(i)));
        oss << (test_order.at(i) - 1) << "\t" << (duration * 1000.f) << "\n";
    }
#endif

    return oss.str();
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
        mShadowArguments.emplace_back(make_shadow_args(arg, stream, &mResource));
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
