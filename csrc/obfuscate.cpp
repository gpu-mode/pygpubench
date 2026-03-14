// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "obfuscate.h"

#include <sys/mman.h>
#include <cstring>
#include <random>
#include <string_view>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <cerrno>
#include <cstdio>

constexpr std::size_t PAGE_SIZE = 4096;

ProtectablePage::ProtectablePage() {
    void* page = mmap(nullptr, PAGE_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    printf("%ld %ld %ld %ld\n", page, slow_hash(page), slow_unhash(slow_hash(page)), slow_hash(slow_unhash((std::uintptr_t)page)));
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