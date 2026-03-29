// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <fcntl.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <string_view>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/landlock.h>
#include <seccomp.h>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <sstream>
#include <string>
#include <vector>

class Fd {

public:
    explicit Fd(int fd) : mFD(fd) {}
    ~Fd() { close(mFD); }

    int fd() { return mFD; }

    // non-copyable, movable
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& o) noexcept : mFD(std::exchange(o.mFD, -1)) {}

private:
    int mFD;
};

struct LandlockFd : Fd {
    explicit LandlockFd(int fd) : Fd(fd) {}
};

static LandlockFd landlock_create_ruleset(
    const struct landlock_ruleset_attr *attr, size_t size, uint32_t flags) {
    const int ret = syscall(__NR_landlock_create_ruleset, attr, size, flags);
    if (ret < 0)
        throw std::system_error(errno, std::system_category(),
                                "landlock_create_ruleset");
    return LandlockFd{ret};
}

static void landlock_add_rule(
    LandlockFd& ruleset, enum landlock_rule_type rule_type,
    const void *rule_attr, uint32_t flags) {
    if (syscall(__NR_landlock_add_rule, ruleset.fd(), rule_type, rule_attr, flags) < 0)
        throw std::system_error(errno, std::system_category(),
                                "landlock_add_rule");
}

static void landlock_restrict_self(LandlockFd& ruleset, uint32_t flags) {
    if (syscall(__NR_landlock_restrict_self, ruleset.fd(), flags) < 0)
        throw std::system_error(errno, std::system_category(),
                                "landlock_restrict_self");
}

static void allow_path(LandlockFd& ruleset, const char *path, uint64_t access) {
    int raw = open(path, O_PATH | O_CLOEXEC);
    if (raw < 0) {
        if (errno == ENOENT) return;
        throw std::system_error(errno, std::system_category(), path);
    }
    Fd fd(raw);

    struct landlock_path_beneath_attr attr = {
        .allowed_access = access,
        .parent_fd      = fd.fd(),
    };
    landlock_add_rule(ruleset, LANDLOCK_RULE_PATH_BENEATH, &attr, 0);
}

void install_landlock() {
    const std::uint64_t RO = LANDLOCK_ACCESS_FS_READ_FILE |
                     LANDLOCK_ACCESS_FS_READ_DIR;

    const std::uint64_t RW = RO                              |
                     LANDLOCK_ACCESS_FS_WRITE_FILE   |
                     LANDLOCK_ACCESS_FS_REMOVE_FILE  |
                     LANDLOCK_ACCESS_FS_REMOVE_DIR   |
                     LANDLOCK_ACCESS_FS_MAKE_REG     |
                     LANDLOCK_ACCESS_FS_MAKE_DIR     |
                     LANDLOCK_ACCESS_FS_MAKE_SYM     |
                     #ifdef LANDLOCK_ACCESS_FS_TRUNCATE
                     LANDLOCK_ACCESS_FS_TRUNCATE     |
                     #endif
                     #ifdef LANDLOCK_ACCESS_FS_REFER
                     LANDLOCK_ACCESS_FS_REFER        |
                     #endif
                     0;

    struct landlock_ruleset_attr ruleset_attr = {
        .handled_access_fs = RW, // everything we handle; unlisted = unrestricted
    };

    LandlockFd ruleset_fd = landlock_create_ruleset(&ruleset_attr, sizeof(ruleset_attr), 0);

    // Read-only: entire filesystem
    allow_path(ruleset_fd, "/", RO);

    // Read-write: /tmp and /dev only
    allow_path(ruleset_fd, "/tmp", RW);
    allow_path(ruleset_fd, "/dev", RW); // needed for /dev/null etc, used e.g., by triton

    // required for landlock
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_NO_NEW_PRIVS)");
    };

    landlock_restrict_self(ruleset_fd, 0);
}


// mseal:
// with mseal we can prevent address regions from being remapped with different memory protection attributes
// In particular, we can prevent making existing executable regions (i.e., loaded libraries) writeable,
// so that attempts to monkeypatch, e.g., cudaEventElapsedTime will fail. Crucially, once sealed,
// even the running process itself cannot unseal that address range or replace it with a new mapping.

#ifndef __NR_mseal
#define __NR_mseal 462  // x86-64
#endif

bool mseal_supported() {
    // Call mseal with a null/zero range - older kernels don't implement this syscall
    // ENOSYS = not implemented; EINVAL = implemented but bad args (expected).
    static bool supported = [] {
        syscall(__NR_mseal, 0, 0, 0UL);
        return errno != ENOSYS;
    }();
    return supported;
}

void mseal(void* addr, size_t len, std::string_view name) {
    if (syscall(__NR_mseal, addr, len, 0UL) < 0) {
        throw std::system_error(errno, std::generic_category(), "mseal: " + std::string(name));
    }
}

// these cannot be sealed
static const std::unordered_set<std::string> excluded_paths = {"[vdso]", "[vvar]", "[vsyscall]"};

void seal_mappings() {
    std::ifstream maps("/proc/self/maps");
    if (!maps)
        throw std::runtime_error("Failed to open /proc/self/maps");

    struct Region { uintptr_t start, end; std::string src; };
    std::vector<Region> to_seal;

    std::string line;
    while (std::getline(maps, line)) {
        std::istringstream ss(line);
        std::string range, perms, offset, dev, inode, path;
        if (!(ss >> range >> perms >> offset >> dev >> inode)) continue;
        ss >> path;

        if (excluded_paths.count(path)) continue;

        bool is_exec     = perms.find('x') != std::string::npos;
        bool is_writable = perms.find('w') != std::string::npos;
        bool is_file_backed = !path.empty() && path[0] == '/';
        // seal anything executable and any file-backed read-only memory (e.g. global offset table)
        bool should_seal = is_exec || (!is_writable && is_file_backed);
        if (!should_seal) continue;

        auto dash = range.find('-');
        if (dash == std::string::npos) continue;

        const uintptr_t start = std::stoull(range.substr(0, dash), nullptr, 16);
        const uintptr_t end   = std::stoull(range.substr(dash + 1), nullptr, 16);
        to_seal.push_back({start, end, line});
    }

    for (auto& r : to_seal) {
        mseal(reinterpret_cast<void*>(r.start), r.end - r.start, r.src);
    }
}

static inline void check_seccomp(int rc, const char* what) {
    if (rc < 0)
        throw std::system_error(-rc, std::generic_category(), what);
}

void setup_seccomp_filter(scmp_filter_ctx ctx) {
    check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(ptrace), 0),
                      "block ptrace");

    check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(prctl), 2,
                     SCMP_A0(SCMP_CMP_EQ, PR_SET_DUMPABLE),
                     SCMP_A1(SCMP_CMP_NE, 0)),
                  "block prctl(SET_DUMPABLE!=0)");

    check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(prctl), 1,
                     SCMP_A0(SCMP_CMP_EQ, PR_SET_SECCOMP)),
                "block prctl(SET_SECCOMP)");

    check_seccomp(seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(prctl), 1,
                     SCMP_A0(SCMP_CMP_EQ, PR_SET_PTRACER)),
                "block prctl(SET_PTRACER)");
}

void install_seccomp_filter() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    if (!ctx) throw std::runtime_error("seccomp_init failed");
    try {
        setup_seccomp_filter(ctx);
    }  catch (...) {
        seccomp_release(ctx);
        throw;
    }

    // Prevent ptrace and /proc/self/mem tampering
    if (prctl(PR_SET_DUMPABLE, 0) < 0) {
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_DUMPABLE)");
    }

    // Prevent gaining privileges (if attacker tries setuid exploits)
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_NO_NEW_PRIVS)");
    };
    // no new executable code pages
    // note: this also prevents thread creating, which breaks torch.compile
    // workaround: run torch.compile once from trusted python code, then the thread already
    //             exists at this point. does not seem reliable, so disabled for now
    // prctl(PR_SET_MDWE, PR_MDWE_REFUSE_EXEC_GAIN, 0, 0, 0);

    int rc = seccomp_load(ctx);
    seccomp_release(ctx);
    check_seccomp(rc, "seccomp_load");
}
