// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "obfuscate.h"

#include <sys/mman.h>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <cerrno>
#include <cstdio>

#include <openssl/evp.h>
#include <openssl/rand.h>

constexpr static std::size_t PAGE_SIZE = 4096;

ObfuscatedHexDigest::ObfuscatedHexDigest(std::pmr::monotonic_buffer_resource* mem) {
    void* page = mem->allocate(PAGE_SIZE, PAGE_SIZE);
    HashedPagePtr = slow_hash(reinterpret_cast<std::uintptr_t>(page));
}

void ObfuscatedHexDigest::allocate(std::size_t size, std::mt19937& rng) {
    if (size > PAGE_SIZE / 2) {
        throw std::runtime_error("target size too big");
    }
    if (this->size() != 0) {
        throw std::runtime_error("already allocated");
    }

    fill_random_hex(reinterpret_cast<void*>(slow_unhash(HashedPagePtr)), PAGE_SIZE, rng);
    const std::uintptr_t max_offset = PAGE_SIZE - size - 1;
    std::uniform_int_distribution<std::uintptr_t> offset_dist(0, max_offset);

    const std::uintptr_t offset = offset_dist(rng);
    HashedOffset = slow_hash(offset);
    HashedLen = slow_hash(size ^ offset);
}

const void* ObfuscatedHexDigest::page_ptr() const {
    return reinterpret_cast<const void*>(slow_unhash(HashedPagePtr));
}

char* ObfuscatedHexDigest::data() {
    return reinterpret_cast<char*>(slow_unhash(HashedPagePtr)) + slow_unhash(HashedOffset);
}

std::size_t ObfuscatedHexDigest::size() const {
    return slow_unhash(HashedLen ^ slow_unhash(HashedOffset));
}

void fill_random_hex(void* target, std::size_t size, std::mt19937& rng) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::uniform_int_distribution<int> hex_dist(0, 15);
    auto* page_bytes = static_cast<char*>(target);
    for (std::size_t i = 0; i < size; i++) {
        page_bytes[i] = hex_chars[hex_dist(rng)];
    }
}

std::uintptr_t slow_hash(std::uintptr_t p, int rounds) {
    for (int i = 0; i < rounds; i++) {
        p ^= p >> 17;
        p *= 0xbf58476d1ce4e5b9ULL;
        p ^= p >> 31;
    }
    return p;
}

std::uintptr_t slow_unhash(std::uintptr_t p, int rounds) {
    // run the inverse rounds in reverse order
    for (int i = 0; i < rounds; i++) {
        p ^= (p >> 31) ^ (p >> 62);
        p *= 0x96de1b173f119089ULL;
        p ^= p >> 17 ^ p >> 34 ^ p >> 51;
    }
    return p;
}

void cleanse(void* ptr, size_t size) {
    OPENSSL_cleanse(ptr, size);
}

std::string encrypt_message(const char* key, size_t keyLen, const std::string& plaintext)
{
    if (keyLen != 32)
        throw std::invalid_argument("encrypt_message: key must be exactly 32 bytes for AES-256");

    constexpr int NONCE_LEN = 12;
    constexpr int TAG_LEN   = 16;

    unsigned char nonce[NONCE_LEN];
    if (RAND_bytes(nonce, NONCE_LEN) != 1)
        throw std::runtime_error("encrypt_message: RAND_bytes failed");

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        throw std::runtime_error("encrypt_message: EVP_CIPHER_CTX_new failed");
    struct CtxGuard { EVP_CIPHER_CTX* c; ~CtxGuard() { EVP_CIPHER_CTX_free(c); } } guard{ctx};

    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, NONCE_LEN, nullptr) != 1 ||
        EVP_EncryptInit_ex(ctx, nullptr, nullptr, reinterpret_cast<const unsigned char*>(key), nonce) != 1)
    {
        throw std::runtime_error("encrypt_message: GCM init failed");
    }

    std::vector<unsigned char> ciphertext(plaintext.size());
    int out_len = 0;
    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len,
                          reinterpret_cast<const unsigned char*>(plaintext.data()),
                          static_cast<int>(plaintext.size())) != 1)
    {
        throw std::runtime_error("encrypt_message: EVP_EncryptUpdate failed");
    }

    int final_len = 0;
    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len) != 1)
        throw std::runtime_error("encrypt_message: EVP_EncryptFinal_ex failed");

    unsigned char tag[TAG_LEN];
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, TAG_LEN, tag) != 1)
        throw std::runtime_error("encrypt_message: GCM get tag failed");

    std::string packet;
    packet.reserve(NONCE_LEN + TAG_LEN + plaintext.size());
    packet.append(reinterpret_cast<char*>(nonce),             NONCE_LEN);
    packet.append(reinterpret_cast<char*>(tag),               TAG_LEN);
    packet.append(reinterpret_cast<char*>(ciphertext.data()), out_len + final_len);

    return packet;
}
