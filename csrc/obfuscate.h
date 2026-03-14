// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PYGPUBENCH_OBFUSCATE_H
#define PYGPUBENCH_OBFUSCATE_H

#include <string_view>
#include <random>

// A single memory page that can be read-protected.
// This does not provide any actual defence against an attacker,
// because they could always just remove memory protection before
// access. But that in itself serves to increase the complexity of
// an attack.
class ProtectablePage {
public:
    ProtectablePage();
    ~ProtectablePage();
    ProtectablePage(ProtectablePage&& other) noexcept;

    void lock();
    void unlock();

    [[nodiscard]] void* page_ptr() const;

    std::uintptr_t Page;
};

class ObfuscatedHexDigest : ProtectablePage {
public:
    ObfuscatedHexDigest() = default;

    void allocate(std::size_t size, std::mt19937& rng);

    char* data();

    [[nodiscard]] std::size_t size() const {
        return Len;
    }
private:
    std::size_t Len = 0;
    std::size_t Offset = 0;
};

void fill_random_hex(void* target, std::size_t size, std::mt19937& rng);

std::uintptr_t slow_hash(std::uintptr_t p, int rounds = 100'000);
std::uintptr_t slow_unhash(std::uintptr_t p, int rounds = 100'000);

template<class T>
std::uintptr_t slow_hash(T* ptr, int rounds = 100'000) {
    return slow_hash(reinterpret_cast<std::uintptr_t>(ptr), rounds);
}

#endif //PYGPUBENCH_OBFUSCATE_H