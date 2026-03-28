#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <system_error>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <unistd.h>

struct RangeMsg { uintptr_t lo, hi; };

static int recv_unotify_fd(int sock, uintptr_t& lo, uintptr_t& hi) {
    RangeMsg range;
    struct iovec iov = { &range, sizeof(range) };

    // Ancillary buffer for one fd
    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } cmsg_buf;

    struct msghdr msg = {};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    ssize_t n = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
    if (n < 0) {
        perror("supervisor: recvmsg");
        return -1;
    }
    if (n != sizeof(range)) {
        fprintf(stderr, "supervisor: short read: %zd\n", n);
        return -1;
    }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS
        || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) {
        fprintf(stderr, "supervisor: missing or malformed SCM_RIGHTS\n");
        return -1;
    }

    int unotify_fd;
    memcpy(&unotify_fd, CMSG_DATA(cmsg), sizeof(int));
    lo = range.lo;
    hi = range.hi;
    return unotify_fd;
}

// Returns true if [addr, addr+size) overlaps [lo, hi).
// Handles wraparound: if addr+size wraps, the range covers everything above addr,
// which necessarily overlaps any [lo, hi) where hi > lo.
static bool overlaps(uintptr_t addr, uintptr_t size, uintptr_t lo, uintptr_t hi) {
    // addr+size > lo  AND  addr < hi
    // For wraparound case (addr+size < addr), the range wraps around the
    // address space, so it definitely overlaps any non-empty [lo, hi).
    uintptr_t end = addr + size;
    bool wrapped = (end < addr);
    return (wrapped || end > lo) && (addr < hi);
}

// mprotect/mmap/munmap/madvise/remap_file_pages: args[0]=addr, args[1]=len.
// mremap: blocked unconditionally — MREMAP_FIXED moves the mapping to a new
// address chosen by the caller, making a safe overlap check impossible.
static bool handle_notification(int unotify_fd, uintptr_t lo, uintptr_t hi) {
    struct seccomp_notif req = {};
    struct seccomp_notif_resp resp = {};

    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_RECV, &req) < 0) {
        if (errno == EINTR) return true;   // interrupted, keep going
        if (errno == ENODEV) return false; // tracee thread exited, we're done
        perror("supervisor: SECCOMP_IOCTL_NOTIF_RECV");
        return false;
    }

    resp.id    = req.id;
    resp.flags = 0;
    resp.error = 0;

    // Check the notification is still valid before we act on it.
    // This closes the race where the thread exits between RECV and SEND.
    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_ID_VALID, &req.id) < 0) {
        // Thread is gone — keep looping in case other threads share this filter.
        return true;
    }

    // All remaining syscalls (mprotect, mmap, munmap, madvise):
    // args[0]=addr, args[1]=len — check for overlap with protected range.
    bool deny = overlaps(req.data.args[0], req.data.args[1], lo, hi);

    if (deny) {
        resp.error = -EPERM;
    } else {
        resp.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
    }

    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_SEND, &resp) < 0) {
        if (errno == ENOENT) return true; // thread gone between ID_VALID and SEND, fine
        perror("supervisor: SECCOMP_IOCTL_NOTIF_SEND");
    }
    return true;
}

// Entry point for the supervisor process.
// sock_fd is the supervisor's end of the socketpair, passed directly from Python.
int supervisor_main(int sock_fd) {
    if (prctl(PR_SET_DUMPABLE, 0) < 0) {
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_DUMPABLE)");
    }

    // Die if our parent (the tracee process) dies.
    prctl(PR_SET_PDEATHSIG, SIGTERM);

    uintptr_t lo, hi;
    int unotify_fd = recv_unotify_fd(sock_fd, lo, hi);
    close(sock_fd);

    if (unotify_fd < 0) return 1;

    // Event loop: handle notifications until the tracee thread exits,
    // at which point the unotify fd becomes invalid and NOTIF_RECV returns ENODEV.
    while (handle_notification(unotify_fd, lo, hi))
        ;

    close(unotify_fd);
    return 0;
}
