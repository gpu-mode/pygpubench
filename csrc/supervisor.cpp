#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <system_error>
#include <vector>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <unistd.h>
#include <syscall.h>

#include "protocol.h"
#include <sys/mman.h>

struct Config {
    uintptr_t sensitive_lo;
    uintptr_t sensitive_hi;
    std::vector<AllowedSite> allowed;
};

static int recv_setup(int sock, Config& cfg) {
    SupervisorSetupMsg setup;
    ssize_t n = recv(sock, &setup, sizeof(setup), MSG_WAITALL);
    if (n != sizeof(setup)) {
        fprintf(stderr, "supervisor: short read on SupervisorSetupMsg\n");
        return -1;
    }

    cfg.sensitive_lo = setup.sensitive_lo;
    cfg.sensitive_hi = setup.sensitive_hi;
    cfg.allowed.resize(setup.n_allowed_sites);

    if (setup.n_allowed_sites > 0) {
        size_t sites_sz = setup.n_allowed_sites * sizeof(AllowedSite);
        n = recv(sock, cfg.allowed.data(), sites_sz, MSG_WAITALL);
        if ((size_t)n != sites_sz) {
            fprintf(stderr, "supervisor: short read on AllowedSite[]\n");
            return -1;
        }
    }

    char dummy;
    struct iovec iov = { &dummy, 1 };

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } cmsg_buf;

    struct msghdr msg = {};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    n = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
    if (n < 0) { perror("supervisor: recvmsg"); return -1; }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type != SCM_RIGHTS ||
        cmsg->cmsg_len != CMSG_LEN(sizeof(int))) {
        fprintf(stderr, "supervisor: missing SCM_RIGHTS\n");
        return -1;
    }

    int unotify_fd;
    memcpy(&unotify_fd, CMSG_DATA(cmsg), sizeof(int));
    return unotify_fd;
}

static bool overlaps(uintptr_t addr, uintptr_t size, uintptr_t lo, uintptr_t hi) {
    uintptr_t end = addr + size;
    bool wrapped = (end < addr);
    return (wrapped || end > lo) && (addr < hi);
}

static bool ip_is_allowed(uintptr_t ip, const Config& cfg) {
    for (AllowedSite site : cfg.allowed) {
        if (ip == site + 2)
            return true;
    }
    return false;
}

static bool handle_notification(int unotify_fd, const Config& cfg) {
    struct seccomp_notif req = {};
    struct seccomp_notif_resp resp = {};

    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_RECV, &req) < 0) {
        if (errno == EINTR)  return true;
        if (errno == ENODEV) return false;
        perror("supervisor: SECCOMP_IOCTL_NOTIF_RECV");
        return false;
    }

    resp.id    = req.id;
    resp.flags = 0;
    resp.error = -EPERM;

    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_ID_VALID, &req.id) < 0)
        return true;

    uintptr_t ip   = req.data.instruction_pointer;
    uintptr_t addr = req.data.args[0];
    uintptr_t len  = req.data.args[1];
    int       prot = (int)req.data.args[2];

    bool ip_ok     = ip_is_allowed(ip, cfg);
    bool contained = (addr >= cfg.sensitive_lo)
                  && (len <= cfg.sensitive_hi - cfg.sensitive_lo)
                  && (addr + len <= cfg.sensitive_hi)
                  && (addr + len >= addr);
    bool prot_safe = prot == PROT_NONE;

    if (!contained) {
        // touches other memory, this is fine
        resp.error = 0;
        resp.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
    } else if ((ip_ok || prot_safe) && req.data.nr == SYS_mprotect) {
        // touches our memory, but either makes it PROT_NONE or is from a whitelisted instruction
        resp.error = 0;
        resp.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
        fprintf(stdout, "Allowed mprotect from ip=0x%lx addr=[0x%lx,0x%lx) prot=%d\n",ip, addr, addr + len, prot);
    } else {
        fprintf(stderr,
            "supervisor: DENIED syscall %d from ip=0x%lx addr=[0x%lx,0x%lx) prot=%d "
            "(ip_ok=%d contained=%d)\n",
             req.data.nr, ip, addr, addr + len, prot, ip_ok, contained);
    }

    if (ioctl(unotify_fd, SECCOMP_IOCTL_NOTIF_SEND, &resp) < 0)
        if (errno != ENOENT)
            perror("supervisor: SECCOMP_IOCTL_NOTIF_SEND");

    return true;
}

int supervisor_main(int sock_fd) {
    if (prctl(PR_SET_DUMPABLE, 0) < 0)
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_DUMPABLE)");

    prctl(PR_SET_PDEATHSIG, SIGTERM);

    Config cfg;
    int unotify_fd = recv_setup(sock_fd, cfg);
    close(sock_fd);

    if (unotify_fd < 0) return 1;

    fprintf(stderr, "supervisor: sensitive=[0x%lx, 0x%lx), %zu allowed sites\n",
            cfg.sensitive_lo, cfg.sensitive_hi, cfg.allowed.size());
    for (AllowedSite site : cfg.allowed)
        fprintf(stderr, "supervisor: allowed site: 0x%lx\n", site);

    while (handle_notification(unotify_fd, cfg))
        ;

    close(unotify_fd);
    return 0;
}
