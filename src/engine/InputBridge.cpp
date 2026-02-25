#include "InputBridge.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>

InputBridge::~InputBridge() { shutdown(); }

bool InputBridge::init(bool creator)
{
    creator_ = creator;
    int flags = creator ? (O_CREAT | O_RDWR) : O_RDWR;
    fd_ = shm_open(HOLOGRAM_SHM_NAME, flags, 0666);
    if (fd_ < 0) {
        perror("InputBridge: shm_open");
        return false;
    }

    if (creator) {
        if (ftruncate(fd_, sizeof(SharedHandData)) < 0) {
            perror("InputBridge: ftruncate");
            close(fd_);
            fd_ = -1;
            return false;
        }
    }

    shm_ = (SharedHandData*)mmap(nullptr, sizeof(SharedHandData),
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, 0);
    if (shm_ == MAP_FAILED) {
        perror("InputBridge: mmap");
        shm_ = nullptr;
        close(fd_);
        fd_ = -1;
        return false;
    }

    if (creator) {
        memset(shm_, 0, sizeof(SharedHandData));
    }

    fprintf(stderr, "InputBridge: shm '%s' %s (fd=%d)\n",
            HOLOGRAM_SHM_NAME, creator ? "created" : "attached", fd_);
    return true;
}

// -----------------------------------------------------------------------
// Seqlock read: wait for even seq, copy all fields, verify seq unchanged
// -----------------------------------------------------------------------
bool InputBridge::read(SharedHandData& out, int maxRetries)
{
    if (!shm_) return false;

    auto* seqPtr = reinterpret_cast<const std::atomic<uint32_t>*>(&shm_->seq);

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        uint32_t s1 = seqPtr->load(std::memory_order_acquire);
        if (s1 & 1u) {
            // Odd = write in progress, spin
            continue;
        }

        // Copy the payload (all fields except seq)
        SharedHandData local;
        local.hand_detected = shm_->hand_detected;
        memcpy(local.lm_x, shm_->lm_x, sizeof(shm_->lm_x));
        memcpy(local.lm_y, shm_->lm_y, sizeof(shm_->lm_y));
        local.timestamp = shm_->timestamp;

        uint32_t s2 = seqPtr->load(std::memory_order_acquire);
        if (s1 == s2) {
            local.seq = s1;
            out = local;
            return true;
        }
        // Sequence changed during copy — retry
    }

    return false;  // Could not get consistent snapshot
}

void InputBridge::shutdown()
{
    if (shm_ && shm_ != MAP_FAILED) {
        munmap(shm_, sizeof(SharedHandData));
        shm_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    // Note: shm_unlink is done by main() at exit, not here
}
