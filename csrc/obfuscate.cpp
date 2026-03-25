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

constexpr std::size_t PAGE_SIZE = 4096;

ProtectablePage::ProtectablePage() {
    void* page = mmap(nullptr, PAGE_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    Page = slow_hash(page);
}

ProtectablePage::~ProtectablePage() {
    void* page = page_ptr();
    if (page) {
        if (mprotect(page, PAGE_SIZE, PROT_READ | PROT_WRITE) != 0) {
            std::perror("mprotect restore failed in ~ProtectablePage");
        }
        if (munmap(page, PAGE_SIZE) != 0) {
            std::perror("munmap failed in ~ProtectablePage");
        }
    }
}

ProtectablePage::ProtectablePage(ProtectablePage&& other) noexcept : Page(std::exchange(other.Page, slow_hash((void*)nullptr))){
}

void ProtectablePage::lock() {
    void* page = page_ptr();
    if (mprotect(page, PAGE_SIZE, PROT_NONE) != 0) {
        throw std::system_error(errno, std::generic_category(), "mprotect(PROT_NONE) failed");
    }
}

void ProtectablePage::unlock() {
    void* page = page_ptr();
    if (mprotect(page, PAGE_SIZE, PROT_READ) != 0) {
        throw std::system_error(errno, std::generic_category(), "mprotect(PROT_READ) failed");
    }
}

void* ProtectablePage::page_ptr() const {
    return reinterpret_cast<void*>(slow_unhash(Page));
}

void ObfuscatedHexDigest::allocate(std::size_t size, std::mt19937& rng) {
    if (size > PAGE_SIZE / 2) {
        throw std::runtime_error("target size too big");
    }
    if (Len != 0 || Offset != 0) {
        throw std::runtime_error("already allocated");
    }

    fill_random_hex(page_ptr(), PAGE_SIZE, rng);
    const std::size_t max_offset = PAGE_SIZE - size - 1;
    std::uniform_int_distribution<std::size_t> offset_dist(0, max_offset);

    Offset = offset_dist(rng);
    Len = size;
}

char* ObfuscatedHexDigest::data() {
    return reinterpret_cast<char*>(page_ptr()) + Offset;
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

std::string encrypt_message(void* key, size_t keyLen, const std::string& plaintext)
{
    if (keyLen != 32)
        throw std::invalid_argument("encrypt_message: key must be exactly 32 bytes for AES-256");

    struct Cleanse
    {
        void* key;
        size_t keyLen;
        ~Cleanse() {
            OPENSSL_cleanse(key, keyLen);
        }
    } cleanse_guard{key, keyLen};

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
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, NONCE_LEN, nullptr)  != 1 ||
        EVP_EncryptInit_ex(ctx, nullptr, nullptr,
                           static_cast<const unsigned char*>(key), nonce)      != 1)
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