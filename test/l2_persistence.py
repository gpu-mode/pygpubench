"""L2-persistence reward-hacking regression test.

A submission can try to inflate its score by keeping a large, *reused* input (weights,
lookup tables, ...) resident in L2 across the benchmark's timed iterations, so the timed
kernel reads it from warm L2 instead of from HBM. It does this by marking the buffer
persisting with a stream access-policy window (``cudaStreamSetAttribute`` /
``cudaStreamAttributeAccessPolicyWindow``, ``hitProp=cudaAccessPropertyPersisting``) backed
by a set-aside L2 carveout (``cudaLimitPersistingL2CacheSize``). Plain streaming writes can
only use the *non*-set-aside portion of L2, so the harness's L2-clearing spam-write cannot
evict the parked lines -- unless the harness first releases the carveout and demotes the
persisting lines (see ``csrc/clear_l2.cu``). On Hopper/Blackwell the default persisting
carveout is already nonzero, so a submission only needs the access-policy window.

The test runs the SAME memory-bound row-wise dot product through the isolated benchmark twice:
  * ``kernel``            -- honest (each iteration reads the weights from a cleared L2).
  * ``kernel_l2_persist`` -- marks the reused weights persisting so they would survive the
                             clear and stay warm across iterations.
With the clear_cache hardening, the persisting variant gets no warm-cache advantage: its
time is the same as the honest one (within noise). The size of the advantage this guards
against scales with the GPU's L2:HBM bandwidth ratio -- large on A100/H100/L4, small on
parts with very fast HBM -- so the assertion is a relative back-to-back comparison on the
same device, and is most discriminating on the L4 used in CI.
"""
import ctypes
import torch
import pygpubench

_M = 2048
_state = None


def _get_state():
    global _state
    if _state is not None:
        return _state

    dev = torch.cuda.current_device()
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    cudart = ctypes.CDLL("libcudart.so")

    # cudaDevAttrMaxPersistingL2CacheSize == 108; fall back to half of L2 if unavailable.
    max_persist = l2 // 2
    v = ctypes.c_int(0)
    if cudart.cudaDeviceGetAttribute(ctypes.byref(v), 108, dev) == 0 and v.value > 0:
        max_persist = v.value

    # Reused weight matrix sized to fit in the persisting carveout (cap 64 MiB).
    w_bytes = min(int(max_persist * 0.75), 64 * 1024 * 1024)
    k = max(1, w_bytes // (4 * _M))
    w = torch.ones(k, _M, device="cuda", dtype=torch.float32).contiguous()

    _state = {
        "cudart": cudart,
        "l2": l2,
        "max_persist": max_persist,
        "k": k,
        "w": w,
    }
    return _state


class _AccessPolicyWindow(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t),
                ("hitRatio", ctypes.c_float), ("hitProp", ctypes.c_int), ("missProp", ctypes.c_int)]


class _StreamAttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", _AccessPolicyWindow), ("_pad", ctypes.c_byte * 64)]


_CUDA_ACCESS_PROPERTY_STREAMING = 1
_CUDA_ACCESS_PROPERTY_PERSISTING = 2
_CUDA_STREAM_ATTR_ACCESS_POLICY_WINDOW = 1
_CUDA_LIMIT_PERSISTING_L2 = 0x05
_installed = False


def _install_persistence():
    """Pin _W in L2 by reserving a carveout and setting a persisting access-policy window."""
    state = _get_state()
    w = state["w"]
    cudart = state["cudart"]
    nbytes = w.numel() * w.element_size()
    want = min(state["max_persist"], nbytes)
    cudart.cudaDeviceSetLimit(ctypes.c_int(_CUDA_LIMIT_PERSISTING_L2), ctypes.c_size_t(want))
    stream = torch.cuda.current_stream().cuda_stream
    val = _StreamAttrValue()
    val.accessPolicyWindow.base_ptr = ctypes.c_void_p(w.data_ptr())
    val.accessPolicyWindow.num_bytes = want
    val.accessPolicyWindow.hitRatio = 1.0
    val.accessPolicyWindow.hitProp = _CUDA_ACCESS_PROPERTY_PERSISTING
    val.accessPolicyWindow.missProp = _CUDA_ACCESS_PROPERTY_STREAMING
    cudart.cudaStreamSetAttribute(ctypes.c_void_p(stream),
                                  ctypes.c_int(_CUDA_STREAM_ATTR_ACCESS_POLICY_WINDOW),
                                  ctypes.byref(val))


def _rowwise_dot(w, x, out=None):
    result = torch.sum(w * x, dim=1)
    if out is None:
        return result.contiguous()
    out.copy_(result)
    return out


def kernel(output, x):
    _rowwise_dot(_get_state()["w"], x, out=output)


def kernel_l2_persist(output, x):
    global _installed
    if not _installed:
        _install_persistence()
        _installed = True
    _rowwise_dot(_get_state()["w"], x, out=output)


def generate_test_case(*, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.rand(_M, device="cuda", dtype=torch.float32, generator=gen).contiguous()
    w = _get_state()["w"]
    y = torch.empty(w.shape[0], device="cuda", dtype=torch.float32).contiguous()
    expected = _rowwise_dot(w, x)
    return (y, x), (expected, 1e-2, 1e-2)


def _bench(qualname, repeats=120, seed=1234):
    # landlock/mseal disabled for portability (e.g. Modal CI); see test/grayscale.py.
    res = pygpubench.do_bench_isolated(qualname, generate_test_case, {}, repeats, seed,
                                       discard=True, landlock=False, mseal=False, allow_root=True)
    assert res.success, f"{qualname}: benchmark did not succeed (errors={res.errors}, full={res.full})"
    return pygpubench.basic_stats(res.time_us)


if __name__ == "__main__":
    state = _get_state()
    w = state["w"]
    w_mib = w.numel() * w.element_size() / (1024 * 1024)
    print(f"weights = {w_mib:.0f} MiB (K={state['k']} x M={_M}), "
          f"max_persist = {state['max_persist']/1048576:.0f} MiB, "
          f"L2 = {state['l2']/1048576:.0f} MiB")

    honest = _bench("l2_persistence.kernel")
    hacked = _bench("l2_persistence.kernel_l2_persist")
    print(f"honest         : {honest}")
    print(f"l2-persistence : {hacked}")

    # With clear_cache hardened, the persistence trick must confer no warm-cache speedup:
    # the attack's median must not beat the honest median beyond run-to-run noise.
    ratio = hacked.median / honest.median
    print(f"hacked/honest median ratio = {ratio:.3f} (require >= 0.88)")
    assert ratio >= 0.88, (
        f"L2-persistence trick gained a warm-cache advantage: hacked median "
        f"{hacked.median:.2f}us vs honest {honest.median:.2f}us (ratio {ratio:.3f}); "
        f"clear_cache failed to evict the persisting buffer between iterations."
    )
    print("OK: L2 persistence confers no benchmark advantage.")
