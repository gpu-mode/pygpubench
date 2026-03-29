#include <cerrno>
#include <cstdint>
#include <cstring>
#include <system_error>

#include <sys/socket.h>
#include <unistd.h>
#include <seccomp.h>

#include "protect.h"
#include "protocol.h"

static inline void check_seccomp(int rc, const char* what) {
    if (rc < 0)
        throw std::system_error(-rc, std::generic_category(), what);
}

static void send_all(int sock, const void* buf, size_t len) {
    const auto* p = static_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = send(sock, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::system_category(), "send");
        }
        p   += n;
        len -= n;
    }
}

// ---------------------------------------------------------------------------
// Install a seccomp filter on the calling thread that sends all memory-range
// syscalls to the supervisor via SCMP_ACT_NOTIFY.
//
// We notify on all six rather than trying to check arg[0] in BPF, because
// the overlap check (does [addr, addr+size) intersect [lo, hi)?) requires
// 64-bit addition of two runtime values which BPF cannot do.
//
// Returns the unotify fd.
// ---------------------------------------------------------------------------
static int install_memory_notify_filter() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    if (!ctx)
        throw std::system_error(errno, std::system_category(), "seccomp_init");

    try {
        // These are blocked unconditionally: mremap with MREMAP_FIXED moves
        // mappings to a caller-chosen address (safe overlap check impossible),
        // and remap_file_pages is deprecated with no legitimate JIT use.
        for (int nr : {SCMP_SYS(mremap), SCMP_SYS(remap_file_pages)}) {
            check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), nr, 0),
                          "seccomp_rule_add(block)");
        }

        // These are forwarded to the supervisor for overlap checking.
        for (int nr : {SCMP_SYS(mprotect), SCMP_SYS(mmap),
                       SCMP_SYS(munmap),   SCMP_SYS(madvise)}) {
            check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_NOTIFY, nr, 0),
                          "seccomp_rule_add(notify)");
        }

        check_seccomp(seccomp_load(ctx), "seccomp_load");
    } catch (...) {
        seccomp_release(ctx);
        throw;
    }

    int unotify_fd = seccomp_notify_fd(ctx);
    seccomp_release(ctx);

    if (unotify_fd < 0)
        throw std::system_error(errno, std::system_category(), "seccomp_notify_fd");

    return unotify_fd;
}

// ---------------------------------------------------------------------------
// Send the unotify fd + range to the supervisor over the socketpair.
// ---------------------------------------------------------------------------

static void send_unotify_fd(int sock, int unotify_fd,
                            uintptr_t sensitive_lo, uintptr_t sensitive_hi) {
    uint32_t n = __stop___allowed_mprotect - __start___allowed_mprotect;
    if (n > MAX_ALLOWED_SITES)
        throw std::runtime_error("too many allowed sites");

    SupervisorSetupMsg hdr { sensitive_lo, sensitive_hi, n };

    // Send header + site array as regular data
    send_all(sock, &hdr, sizeof(hdr));

    size_t sites_sz = n * sizeof(AllowedSite);
    send_all(sock, __start___allowed_mprotect, sites_sz);

    // Send unotify_fd via SCM_RIGHTS
    char dummy = 0;
    struct iovec iov = { &dummy, 1 };
    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } cmsg_buf;
    memset(cmsg_buf.buf, 0, sizeof(cmsg_buf.buf));

    struct msghdr msg = {};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &unotify_fd, sizeof(int));

    if (sendmsg(sock, &msg, MSG_NOSIGNAL) < 0)
        throw std::system_error(errno, std::system_category(), "sendmsg unotify_fd");
}

// ---------------------------------------------------------------------------
// Public API: called from the inner (untrusted) thread.
// ---------------------------------------------------------------------------

void seccomp_install_memory_notify(int supervisor_sock, uintptr_t lo, uintptr_t hi) {
    int unotify_fd = install_memory_notify_filter();
    try {
        send_unotify_fd(supervisor_sock, unotify_fd, lo, hi);
    } catch (...) {
        close(unotify_fd);
        throw;
    }
    close(unotify_fd);  // supervisor now owns it; we must not retain it
}


bool supports_seccomp_notify() {
    return seccomp_api_get() >= 5;
}
