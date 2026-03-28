// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <unistd.h>
#include <sys/syscall.h>
#include <system_error>

// Generates an inline mprotect syscall, rounding ptr..ptr+size out to page boundaries,
// and registers the exact address of the syscall instruction in the linker section
// __allowed_mprotect. This section is read at startup to tell the seccomp supervisor
// which instruction pointers are permitted to issue mprotect.
//
// The registration works as follows: the GNU assembler numeric label "1:" is placed
// directly on the syscall instruction in .text. A .quad 1f (a forward reference to
// that label) is emitted into __allowed_mprotect before switching back to .text.
// Because numeric labels are reusable, each expansion of this macro gets its own
// independent "1:" with no clashes, making it safe to use at multiple call sites.
//
#define PROTECT_RANGE(ptr, size, prot)                                        \
    do {                                                                      \
        const uintptr_t _page = static_cast<uintptr_t>(getpagesize());        \
        uintptr_t _start = reinterpret_cast<uintptr_t>(ptr) & ~(_page - 1);   \
        uintptr_t _end   = (reinterpret_cast<uintptr_t>(ptr)                  \
                           + static_cast<uintptr_t>(size) + _page - 1)        \
                           & ~(_page - 1);                                    \
        long _ret;                                                            \
        asm volatile (                                                        \
            ".pushsection __allowed_mprotect, \"aw\"\n\t"                     \
            ".quad 1f\n\t"                                                    \
            ".popsection\n\t"                                                 \
            "1: syscall\n\t"                                                  \
            : "=a"(_ret)                                                      \
            : "0"(__NR_mprotect),                                             \
              "D"(_start),                                                    \
              "S"(_end - _start),                                             \
              "d"(static_cast<long>(prot))                                    \
            : "rcx", "r11", "memory"                                          \
        );                                                                    \
        if (_ret < 0)                                                         \
            throw std::system_error(                                          \
                static_cast<int>(-_ret),                                      \
                std::system_category(), "mprotect");                          \
    } while(0)

extern unsigned long __start___allowed[];
extern unsigned long __stop___allowed[];
