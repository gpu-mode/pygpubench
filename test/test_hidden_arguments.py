"""Regression test for issue #3, hide the benchmark arguments from the python stack.

The compiled ``do_bench`` imports and runs the user kernel on a second thread,
``run_thread`` in ``csrc/binding.cpp``, while ``_do_bench_impl`` is still on the
stack of the main thread. ``sys._current_frames()`` returns that frame, and one
frame further down ``BaseProcess.run`` holds ``self``, whose ``_args`` is the
tuple the child was started with. Before the fix both held the result pipe and
the expected output generator.

The tests stub the compiled extension with a probe that runs the frame walk on
a worker thread, the way the kernel would. They run on CPU and need no GPU. One
test does it in a real spawned child, the others call ``_do_bench_impl``
through a stand in for the child Process object so the ``run`` frame and
``_args`` are on the stack the same way.
"""

import json
import multiprocessing as mp
import multiprocessing.connection
import multiprocessing.process
import os
import sys
import threading
import types
import unittest

# Import the package from source when the test runs from a checkout, and
# register a stub for the compiled extension before the package is imported.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_TEST_DIR), "python")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_stub = types.ModuleType("pygpubench._pygpubench")


def _unset_do_bench(*args, **kwargs):
    raise AssertionError("each test installs its own do_bench probe")


_stub.do_bench = _unset_do_bench
sys.modules["pygpubench._pygpubench"] = _stub

# utils.py imports AESGCM at import time. It is a real dependency of the package,
# but nothing here touches the crypto path, so provide a placeholder when it is
# missing. An environment with cryptography installed uses it as normal.
try:
    import cryptography  # noqa: F401
except ImportError:
    _aead = types.ModuleType("cryptography.hazmat.primitives.ciphers.aead")
    _aead.AESGCM = object
    for _name in (
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.ciphers",
        "cryptography.hazmat.primitives.ciphers.aead",
    ):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    sys.modules["cryptography.hazmat.primitives.ciphers.aead"] = _aead

import pygpubench


class _NoOpDeterministicContext:
    """Stand in for DeterministicContext so the test does not need torch."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return False


# The argument names an imported kernel would look for on the harness frame.
_SECRET_NAMES = ("out_fd", "in_fd", "supervisor_sock", "test_generator", "test_args")


def _a_generator(**kwargs):
    return (), ()


def _is_connection(value):
    return isinstance(value, multiprocessing.connection.Connection)


def _walk_all_threads(secret_ids, skip_code):
    """Walk every frame of every thread through sys._current_frames.

    Returns the secret names found in frame locals, and the places where a
    secret object was found by identity, as a local, inside a tuple or list, or
    inside the _args of a Process like object. Frames running skip_code model
    the C++ side, which holds the arguments and is not a python frame. Module
    frames are skipped, their locals are the module globals, which is where the
    generator used by the spawn test has to live so it can be pickled.
    """
    names = set()
    places = []
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code is not skip_code and frame.f_locals is not frame.f_globals:
                for name, value in frame.f_locals.items():
                    if name in _SECRET_NAMES:
                        names.add(name)
                    if id(value) in secret_ids or _is_connection(value):
                        places.append((frame.f_code.co_name, name))
                    if isinstance(value, (tuple, list)):
                        for item in value:
                            if id(item) in secret_ids or _is_connection(item):
                                places.append((frame.f_code.co_name, name + "[]"))
                    if hasattr(value, "_target") and hasattr(value, "_args"):
                        for item in value._args:
                            if id(item) in secret_ids or _is_connection(item):
                                places.append((frame.f_code.co_name, name + "._args"))
            frame = frame.f_back
    return names, places


class _Probe:
    """Models the compiled do_bench. Runs the walk on a worker thread, records
    what that thread can still reach, and writes to the result descriptor. With
    report set it also writes its findings as json, for the spawned child."""

    def __init__(self, report=False):
        self.report = report
        self.names = None
        self.places = None
        self.wrote = None
        self.inheritable = None
        self.args_cleared = None

    def __call__(self, *c_args):
        result_fd = c_args[0]
        secret_ids = {id(c_args[4]), id(c_args[5])}
        self.inheritable = os.get_inheritable(result_fd)
        self.args_cleared = getattr(mp.current_process(), "_args", ()) == ()

        def user_code():
            self.names, self.places = _walk_all_threads(secret_ids, _Probe.__call__.__code__)
            self.wrote = os.write(result_fd, b"probe-wrote")

        worker = threading.Thread(target=user_code)
        worker.start()
        worker.join()
        if self.report:
            findings = {
                "names": sorted(self.names),
                "places": self.places,
                "wrote": self.wrote,
                "inheritable": self.inheritable,
                "args_cleared": self.args_cleared,
            }
            os.write(result_fd, json.dumps(findings).encode())


class _FakeProcess:
    """The child Process object as the spawned child sees it. run() binds self on
    the frame below the target, the way BaseProcess.run does."""

    def __init__(self, target, args):
        self._target = target
        self._args = args
        self._kwargs = {}

    def run(self):
        saved = multiprocessing.process._current_process
        multiprocessing.process._current_process = self
        try:
            self._target(*self._args, **self._kwargs)
        finally:
            multiprocessing.process._current_process = saved


def _vulnerable_impl(out_fd, in_fd, supervisor_sock, qualname, test_generator, test_args,
                     stream=None, discard=True, nvtx=False, tb_conn=None, landlock=True,
                     mseal=True, allow_root=False):
    """The shape of _do_bench_impl before the fix, kept as a control so the test
    fails if the probe stops seeing a leak."""
    pygpubench._pygpubench.do_bench(
        out_fd.fileno(), in_fd.fileno(), supervisor_sock.fileno(), qualname,
        test_generator, test_args, stream, discard, nvtx, landlock, mseal, allow_root,
    )


def _spawn_entry(*packed):
    """Target of the real spawned child. Installs the reporting probe and calls
    the real _do_bench_impl. The call is handed over by popping it, so this frame
    keeps no reference to the arguments while the probe walks the stack."""
    pygpubench.DeterministicContext = _NoOpDeterministicContext
    pygpubench._pygpubench.do_bench = _Probe(report=True)
    call = [packed]
    del packed
    pygpubench._do_bench_impl(*call.pop())


class HiddenArgumentsTest(unittest.TestCase):
    def setUp(self):
        self._saved_ctx = pygpubench.DeterministicContext
        pygpubench.DeterministicContext = _NoOpDeterministicContext
        self._saved_do_bench = pygpubench._pygpubench.do_bench
        # Names in this frame stay clear of the secret names, or the walk would
        # find them here and report a false leak.
        self._read_end, self._write_end = mp.Pipe(duplex=False)
        self._sig_r, self._sig_w = mp.Pipe(duplex=False)
        self._sup_a, self._sup_b = mp.Pipe(duplex=False)
        self._generator = _a_generator
        self._arguments = {"size": 256}

    def tearDown(self):
        pygpubench.DeterministicContext = self._saved_ctx
        pygpubench._pygpubench.do_bench = self._saved_do_bench
        for conn in (self._read_end, self._write_end, self._sig_r,
                     self._sig_w, self._sup_a, self._sup_b):
            try:
                conn.close()
            except OSError:
                pass

    def _packed_args(self):
        # (out_fd, in_fd, supervisor_sock, qualname, test_generator, test_args,
        #  stream, discard, nvtx, tb_conn, landlock, mseal, allow_root)
        return (
            self._write_end, self._sig_r, self._sup_b, "some_module.kernel",
            self._generator, self._arguments, 0, True, False, None, True, True, False,
        )

    def _probe(self):
        probe = _Probe()
        pygpubench._pygpubench.do_bench = probe
        return probe

    def test_control_leaks_arguments(self):
        # The probe must see the leak in the old shape, otherwise the fix proves nothing.
        probe = self._probe()
        _FakeProcess(_vulnerable_impl, self._packed_args()).run()
        self.assertEqual(probe.wrote, len(b"probe-wrote"))
        self.assertEqual(probe.names, set(_SECRET_NAMES))
        found = {place for _, place in probe.places}
        self.assertIn("test_generator", found)
        self.assertIn("out_fd", found)
        self.assertIn("self._args", found)

    def test_fix_hides_arguments_from_frame_walk(self):
        probe = self._probe()
        _FakeProcess(pygpubench._do_bench_impl, self._packed_args()).run()
        self.assertEqual(probe.names, set(), "a secret name is still bound on a reachable frame")
        self.assertEqual(probe.places, [], "a secret object is still reachable from a frame")
        self.assertTrue(probe.args_cleared)

    def test_result_pipe_still_writable(self):
        probe = self._probe()
        _FakeProcess(pygpubench._do_bench_impl, self._packed_args()).run()
        self.assertEqual(probe.wrote, len(b"probe-wrote"))
        self.assertTrue(self._read_end.poll(2), "the probe should have written to the pipe")
        self.assertEqual(os.read(self._read_end.fileno(), 64), b"probe-wrote")
        # The Connection objects are closed, the runner wrote through its own descriptor.
        self.assertTrue(self._write_end.closed)
        self.assertTrue(self._sig_r.closed)
        self.assertTrue(self._sup_b.closed)

    def test_runner_descriptors_are_close_on_exec(self):
        # Under spawn the pipes arrive inheritable. Model that, then check the
        # descriptor the runner receives has FD_CLOEXEC set.
        os.set_inheritable(self._write_end.fileno(), True)
        os.set_inheritable(self._sig_r.fileno(), True)
        probe = self._probe()
        _FakeProcess(pygpubench._do_bench_impl, self._packed_args()).run()
        self.assertFalse(probe.inheritable)

    def test_real_spawned_child(self):
        ctx = mp.get_context("spawn")
        read_end, write_end = ctx.Pipe(duplex=False)
        sig_r, sig_w = ctx.Pipe(duplex=False)
        sup_a, sup_b = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_spawn_entry,
                            args=(write_end, sig_r, sup_b, "some_module.kernel", _a_generator,
                                  {"size": 256}, 0, True, False, None, True, True, False))
        child.start()
        write_end.close()
        sig_r.close()
        sup_b.close()
        chunks = b""
        while True:
            chunk = os.read(read_end.fileno(), 65536)
            if chunk == b"":
                break
            chunks += chunk
        child.join(60)
        for conn in (read_end, sig_w, sup_a):
            conn.close()
        self.assertEqual(child.exitcode, 0)
        self.assertTrue(chunks.startswith(b"probe-wrote"), chunks)
        findings = json.loads(chunks[len(b"probe-wrote"):])
        self.assertEqual(findings["names"], [])
        self.assertEqual(findings["places"], [])
        self.assertEqual(findings["wrote"], len(b"probe-wrote"))
        self.assertFalse(findings["inheritable"])
        self.assertTrue(findings["args_cleared"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
