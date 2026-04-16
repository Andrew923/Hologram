#include "DNAHelixApp.h"
#include "VoxelPaint.h"
#include "GestureDetector.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr int   N_STEPS    = 64;     // samples along the helix
static constexpr int   BASE_STEP  = 6;      // rung every N-th sample
static constexpr float HELIX_TURNS   = 2.5f;
static constexpr float HELIX_RADIUS  = 0.55f;
static constexpr float HELIX_HEIGHT  = 1.6f;

// Scale mapping (same as CubeApp).
static constexpr float SCALE_MIN_PX = 10.0f;
static constexpr float SCALE_MAX_PX = 26.0f;
static constexpr float SCALE_MIN = SCALE_MIN_PX / (VOXEL_H - 1);
static constexpr float SCALE_MAX = SCALE_MAX_PX / (VOXEL_H - 1);

static constexpr float SMOOTHING        = 0.15f;
static constexpr float ANGULAR_GAIN     = 0.12f;  // radians/frame per unit dir
static constexpr float ANGULAR_VEL_DECAY = 0.92f; // no gesture → velocity fades

// Colors
static const uint8_t STRAND_A_RGB[3] = {255, 40, 200};  // magenta
static const uint8_t STRAND_B_RGB[3] = {0,   220, 255}; // cyan
static const uint8_t RUNG1_RGB[3]    = {255, 220, 0};   // yellow
static const uint8_t RUNG2_RGB[3]    = {0,   255, 60};  // green

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void DNAHelixApp::setup(Renderer& /*renderer*/) {}

void DNAHelixApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    if (!hand.hand_detected) {
        angularVel_ *= ANGULAR_VEL_DECAY;
        phase_ += angularVel_;
        return;
    }

    Gesture g = detectGesture(hand);

    // PEACE → two-finger rotation control.
    if (g == Gesture::PEACE) {
        // Direction of the V: index tip(8) minus index MCP(5).
        float dirX = hand.lm_x[8] - hand.lm_x[5];
        // Clamp to a reasonable range and scale to angular velocity.
        dirX = clampf(dirX, -0.5f, 0.5f);
        float target = dirX * 2.0f * ANGULAR_GAIN;  // 2.0f restores range
        angularVel_ += (target - angularVel_) * SMOOTHING;
    } else {
        angularVel_ *= ANGULAR_VEL_DECAY;
    }

    // PINCH → scale (same formula as CubeApp::update:136-139).
    if (g == Gesture::PINCH) {
        float pinch = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
        float tgtScale = SCALE_MIN
            + clampf((pinch - 0.03f) / 0.22f, 0.0f, 1.0f)
              * (SCALE_MAX - SCALE_MIN);
        scale_ += (tgtScale - scale_) * SMOOTHING;
        scale_  = clampf(scale_, SCALE_MIN, SCALE_MAX);
    }

    phase_ += angularVel_;
}

void DNAHelixApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    // Precompute sample positions on both strands in model space, then
    // rotate about Y by phase_, scale, and map to voxel space.
    float prevAX = 0, prevAY = 0, prevAZ = 0;
    float prevBX = 0, prevBY = 0, prevBZ = 0;

    for (int i = 0; i < N_STEPS; ++i) {
        float t = (float)i / (float)(N_STEPS - 1);
        float y = (t - 0.5f) * HELIX_HEIGHT;
        float theta = 2.0f * (float)M_PI * HELIX_TURNS * t + phase_;

        // Strand A
        float ax = HELIX_RADIUS * cosf(theta);
        float az = HELIX_RADIUS * sinf(theta);
        // Strand B (opposite side of helix)
        float bx = HELIX_RADIUS * cosf(theta + (float)M_PI);
        float bz = HELIX_RADIUS * sinf(theta + (float)M_PI);

        // Scale (all axes) and map to voxel space. The helix runs along Y.
        auto toVoxel = [](float x, float y, float z, float s,
                          int& vx, int& vy, int& vz) {
            vx = (int)roundf((x * s + 1.0f) * 0.5f * (VOXEL_W - 1));
            vy = (int)roundf((y * s + 1.0f) * 0.5f * (VOXEL_H - 1));
            vz = (int)roundf((z * s + 1.0f) * 0.5f * (VOXEL_D - 1));
        };

        int vax, vay, vaz, vbx, vby, vbz;
        toVoxel(ax, y, az, scale_, vax, vay, vaz);
        toVoxel(bx, y, bz, scale_, vbx, vby, vbz);

        // Paint strand segments (connect to the previous sample).
        if (i > 0) {
            int pvax = (int)roundf((prevAX * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1));
            int pvay = (int)roundf((prevAY * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1));
            int pvaz = (int)roundf((prevAZ * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1));
            int pvbx = (int)roundf((prevBX * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1));
            int pvby = (int)roundf((prevBY * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1));
            int pvbz = (int)roundf((prevBZ * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1));

            voxpaint::paint3DLine(voxels, pvax, pvay, pvaz, vax, vay, vaz,
                                  STRAND_A_RGB[0], STRAND_A_RGB[1], STRAND_A_RGB[2]);
            voxpaint::paint3DLine(voxels, pvbx, pvby, pvbz, vbx, vby, vbz,
                                  STRAND_B_RGB[0], STRAND_B_RGB[1], STRAND_B_RGB[2]);
        }

        // Paint base-pair rungs at regular intervals, alternating colors.
        if (i % BASE_STEP == 0) {
            const uint8_t* rc = ((i / BASE_STEP) % 2 == 0) ? RUNG1_RGB : RUNG2_RGB;
            voxpaint::paint3DLine(voxels, vax, vay, vaz, vbx, vby, vbz,
                                  rc[0], rc[1], rc[2]);
        }

        prevAX = ax; prevAY = y; prevAZ = az;
        prevBX = bx; prevBY = y; prevBZ = bz;
    }

    renderer.uploadVoxelBuffer(voxels);
}
