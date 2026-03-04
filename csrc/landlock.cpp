// Copyright (c) 2026 Erik Schultheis
// All rights reserved.
//

#include <cstdint>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/landlock.h>
#include <system_error>
#include <utility>

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

    // Prevent ptrace and /proc/self/mem tampering
    prctl(PR_SET_DUMPABLE, 0);
    // Prevent gaining privileges (if attacker tries setuid exploits)
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
        throw std::system_error(errno, std::system_category(), "prctl(PR_SET_NO_NEW_PRIVS)");
    };
    // no new executable code pages
    // note: this also prevents thread creating, which breaks torch.compile
    // workaround: run torch.compile once from trusted python code, then the thread already
    //             exists at this point. does not seem reliable, so disabled for now
    // prctl(PR_SET_MDWE, PR_MDWE_REFUSE_EXEC_GAIN, 0, 0, 0);

    landlock_restrict_self(ruleset_fd, 0);
}
