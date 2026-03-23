import socket
import multiprocessing as mp
from typing import Optional

from . import _pygpubench


def _supervisor_entry(sock: socket.socket) -> None:
    """
    Entry point for the supervisor process.
    Calls into C++ immediately and never returns to Python.
    sock is the supervisor's end of the socketpair.
    """
    _pygpubench.run_supervisor(sock.fileno())
    # run_supervisor loops until the tracee's unotify fd closes, then returns.


class SeccompSupervisor:
    """
    Owns the supervisor process and the tracee-side socket.
    One instance per BenchmarkManager.

    The supervisor process:
      - is spawned fresh via mp.spawn (no CUDA state)
      - receives the seccomp unotify fd + protected range from the inner thread
        via the socket, using SCM_RIGHTS
      - loops handling mprotect notifications until the inner thread exits

    The tracee side:
      - calls seccomp_install_mprotect_notify(tracee_sock_fd, lo, hi)
        from the inner thread, which installs the filter and sends the unotify fd
    """

    def __init__(self) -> None:
        ctx = mp.get_context('spawn')

        # SOCK_SEQPACKET: message-boundary preserving, connection-oriented.
        # If either end dies, the other gets an error immediately.
        supervisor_sock, tracee_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )

        try:
            self._process = ctx.Process(
                target=_supervisor_entry,
                args=(supervisor_sock,),
                daemon=True,  # dies automatically if parent dies
            )
            self._process.start()
        except:
            supervisor_sock.close()
            tracee_sock.close()
            raise
        finally:
            # Parent closes supervisor end immediately after spawn.
            # The spawned process has its own copy.
            supervisor_sock.close()

        self._tracee_sock = tracee_sock

    @property
    def tracee_sock(self) -> socket.socket:
        """Socket to pass to the tracee process via ctx.Process args.
        Survives pickling across mp.spawn via mp.reduction."""
        return self._tracee_sock

    def close(self) -> None:
        """
        Close the tracee-side socket and wait for the supervisor to exit.
        Closing the socket causes the supervisor's unotify fd to become
        invalid, which terminates its event loop.
        """
        if self._tracee_sock:
            self._tracee_sock.close()
            self._tracee_sock = None
        if self._process and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
