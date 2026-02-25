#pragma once
#include <stdint.h>
#include <stddef.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t  seq;              // offset 0  — seqlock (odd=writing, even=ready)
    uint8_t   hand_detected;   // offset 4
    uint8_t   _pad[3];         // offset 5  — explicit padding (do not remove)
    float     lm_x[21];        // offset 8  — MediaPipe X coords [0,1]
    float     lm_y[21];        // offset 92 — MediaPipe Y coords [0,1]
    double    timestamp;       // offset 176
} SharedHandData;              // total: 184 bytes
#pragma pack(pop)

#define HOLOGRAM_SHM_NAME "/hologram_hand"
#define NUM_LANDMARKS 21

// Static offset checks (verified at compile time)
static_assert(offsetof(SharedHandData, seq)           ==   0, "seq offset");
static_assert(offsetof(SharedHandData, hand_detected) ==   4, "hand_detected offset");
static_assert(offsetof(SharedHandData, _pad)          ==   5, "_pad offset");
static_assert(offsetof(SharedHandData, lm_x)          ==   8, "lm_x offset");
static_assert(offsetof(SharedHandData, lm_y)          ==  92, "lm_y offset");
static_assert(offsetof(SharedHandData, timestamp)     == 176, "timestamp offset");
static_assert(sizeof(SharedHandData)                  == 184, "SharedHandData size");
