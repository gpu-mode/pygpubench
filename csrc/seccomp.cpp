#include <sys/prctl.h>
#include <sys/syscall.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <unistd.h>
#include <vector>
#include <cstdint>
#include <cerrno>
#include <system_error>

// Utility to help build up non-trivial bpf for checking address ranges of pointer arguments
struct BpfBuilder {
    std::vector<sock_filter> Instructions;

    // Raw append
    void emit(sock_filter f) { Instructions.push_back(f); }

    size_t size() const { return Instructions.size(); }

    // Reserve a jump slot, return its index so it can patched later with `patch_jf_to_here`
    size_t emit_jump_placeholder(int comp, uint32_t k) {
        size_t idx = Instructions.size();
        Instructions.push_back(BPF_JUMP(BPF_JMP|comp|BPF_K, k, 0, 0));
        return idx;
    }

    // Patch a previously reserved jump's jf to skip to current position.
    void patch_jf_to_here(size_t idx) {
        // jf is relative to the instruction *after* the jump
        Instructions[idx].jf = (uint8_t)(Instructions.size() - idx - 1);
    }

    void patch_jt_to_here(size_t idx) {
        Instructions[idx].jt = (uint8_t)(Instructions.size() - idx - 1);
    }

    // Load a field from struct seccomp_data
    void load(uint32_t offset) {
        emit(BPF_STMT(BPF_LD|BPF_W|BPF_ABS, offset));
    }

    void ret(uint32_t val) {
        emit(BPF_STMT(BPF_RET|BPF_K, val));
    }

    void ret_allow() { ret(SECCOMP_RET_ALLOW); }
    void ret_kill()  { ret(SECCOMP_RET_KILL_PROCESS); }
    void ret_errno(int e) { ret(SECCOMP_RET_ERRNO | (e & SECCOMP_RET_DATA)); }

    // if (loaded_value == k) skip n instructions, else continue
    void jeq_skip(uint32_t k, uint8_t skip) {
        emit(BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K, k, skip, 0));
    }
    // if (loaded_value > k) skip n, else continue
    void jgt_skip(uint32_t k, uint8_t skip) {
        emit(BPF_JUMP(BPF_JMP|BPF_JGT|BPF_K, k, skip, 0));
    }
    // if (loaded_value >= k) skip n, else continue
    void jge_skip(uint32_t k, uint8_t skip) {
        emit(BPF_JUMP(BPF_JMP|BPF_JGE|BPF_K, k, skip, 0));
    }
    // if (loaded_value != k) skip n, else continue
    void jne_skip(uint32_t k, uint8_t skip) {
        emit(BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K, k, 0, skip));
    }

    // Emits instructions that return `on_match` if args[arg_idx] falls within
    // [lo, hi), then fall through otherwise.
    // Leaves the accumulator in an undefined state after.
    void emit_u64_arg_in_range(int arg_idx, uintptr_t lo, uintptr_t hi,
                               uint32_t on_match) {
        uint32_t lo_hi = (uint32_t)(lo >> 32),  lo_lo = (uint32_t)lo;
        uint32_t hi_hi = (uint32_t)(hi >> 32),  hi_lo = (uint32_t)hi;
        uint32_t arg_off = offsetof(struct seccomp_data, args) + arg_idx * sizeof(uint64_t);

        std::vector<size_t> not_in_range_jt; // JGT, JGE: true branch is not-in-range
        std::vector<size_t> not_in_range_jf; // JEQ used as JNE, JGE used as JLT: false branch is not-in-range

        // lower bound
        load(arg_off + 4);
        size_t jgt_lo = emit_jump_placeholder(BPF_JGT, lo_hi);
        not_in_range_jf.push_back(emit_jump_placeholder(BPF_JEQ, lo_hi)); // jf fires when !=
        load(arg_off);
        not_in_range_jf.push_back(emit_jump_placeholder(BPF_JGE, lo_lo)); // jf fires when
        patch_jt_to_here(jgt_lo);

        // upper bound
        load(arg_off + 4);
        not_in_range_jt.push_back(emit_jump_placeholder(BPF_JGT, hi_hi));
        size_t jeq_hi = emit_jump_placeholder(BPF_JEQ, hi_hi);
        ret(on_match);
        patch_jt_to_here(jeq_hi);
        load(arg_off);
        not_in_range_jt.push_back(emit_jump_placeholder(BPF_JGE, hi_lo));
        ret(on_match);

        for (size_t idx : not_in_range_jt) patch_jt_to_here(idx);
        for (size_t idx : not_in_range_jf) patch_jf_to_here(idx);
    }

    // block a syscall if arg is in [lo, hi)
    void block_syscall_if_arg_in_range(int syscall_nr, int arg_idx,
                                       uintptr_t lo, uintptr_t hi) {
        // Load nr; if not this syscall, skip the whole range check.
        load(offsetof(struct seccomp_data, nr));
        size_t skip_idx = emit_jump_placeholder(BPF_JEQ, syscall_nr);

        emit_u64_arg_in_range(arg_idx, lo, hi, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA));

        patch_jf_to_here(skip_idx);
    }

    struct sock_fprog build() {
        return { .len = (unsigned short)Instructions.size(), .filter = Instructions.data() };
    }
};

// ── Public API ───────────────────────────────────────────────────────────────

void seccomp_protect_page_range(uintptr_t protected_page, size_t page_size) {
    BpfBuilder b;

    // Reject wrong arch to prevent syscall number confusion attacks
    b.load(offsetof(struct seccomp_data, arch));
    b.jeq_skip(AUDIT_ARCH_X86_64, 1);
    b.ret_kill();

    uintptr_t lo = protected_page;
    uintptr_t hi = protected_page + page_size;

    // prevent messing with the protected page range
    b.block_syscall_if_arg_in_range(__NR_mprotect,        0, lo, hi);
    b.block_syscall_if_arg_in_range(__NR_mmap,            0, lo, hi);
    b.block_syscall_if_arg_in_range(__NR_mremap,          0, lo, hi);
    b.block_syscall_if_arg_in_range(__NR_munmap,          0, lo, hi);
    b.block_syscall_if_arg_in_range(__NR_madvise,         0, lo, hi);
    b.block_syscall_if_arg_in_range(__NR_remap_file_pages,0, lo, hi);

    b.ret_allow();

    auto prog = b.build();
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        throw std::system_error(errno, std::generic_category(), "prctl(PR_SET_NO_NEW_PRIVS) failed");
    }
    if (syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) != 0) {
        throw std::system_error(errno, std::generic_category(), "seccomp(SECCOMP_SET_MODE_FILTER) failed");
    }
}
