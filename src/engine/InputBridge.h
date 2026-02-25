#pragma once
#include "../../shared_defs.h"
#include <atomic>
#include <cstddef>

class InputBridge {
public:
    InputBridge() = default;
    ~InputBridge();

    // Open (and optionally create+zero) the shared memory segment.
    // creator=true: shm_open O_CREAT|O_RDWR + ftruncate + memset(0)
    // creator=false: shm_open O_RDWR (attach to existing)
    bool init(bool creator = true);

    // Seqlock read with up to maxRetries attempts.
    // Returns true if a consistent snapshot was obtained.
    bool read(SharedHandData& out, int maxRetries = 16);

    void shutdown();

private:
    SharedHandData* shm_ = nullptr;
    int             fd_  = -1;
    bool            creator_ = false;
};
