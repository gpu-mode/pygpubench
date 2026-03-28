// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PYGPUBENCH_OBFUSCATE_H
#define PYGPUBENCH_OBFUSCATE_H

#include <string>
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

    [[nodiscard]] std::size_t size() const;

private:
    std::uintptr_t HashedLen = 0;
    std::uintptr_t HashedOffset = 0;
};

void fill_random_hex(void* target, std::size_t size, std::mt19937& rng);

std::uintptr_t slow_hash(std::uintptr_t p, int rounds = 100'000);
std::uintptr_t slow_unhash(std::uintptr_t p, int rounds = 100'000);

template<class T>
std::uintptr_t slow_hash(T* ptr, int rounds = 100'000) {
    return slow_hash(reinterpret_cast<std::uintptr_t>(ptr), rounds);
}

// Encrypts `plaintext` with AES-256-GCM using `key` (must be exactly 32 bytes).
// Returns a binary packet: [nonce (12)] [tag (16)] [ciphertext (N)].
// key will be cleansed after use
std::string encrypt_message(void* key, size_t keyLen, const std::string& plaintext);

#endif //PYGPUBENCH_OBFUSCATE_H