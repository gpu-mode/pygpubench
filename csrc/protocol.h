// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

// Wire protocol between the benchmark process and the seccomp supervisor.
// Both sides include this header; neither side should define these structs independently.

#pragma once
#include <cstdint>

constexpr int MAX_ALLOWED_SITES = 32;

// Sent as regular data before the SCM_RIGHTS message.
// Followed immediately by n_allowed_sites * sizeof(uintptr_t) bytes,
// each being the exact address of an allowed mprotect syscall instruction.
struct SupervisorSetupMsg {
    std::uintptr_t sensitive_lo;      // protected arena start
    std::uintptr_t sensitive_hi;      // protected arena end (exclusive)
    std::uint32_t  n_allowed_sites;   // number of entries that follow
};

// Each allowed site is a single pointer — the address of a `syscall` instruction
// registered via PROTECT_RANGE. The supervisor allows mprotect only when the
// instruction pointer is at site+2 (instruction past the syscall?)
using AllowedSite = std::uintptr_t;
