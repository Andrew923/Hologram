#include "TorusKnotApp.h"
#include "VoxelPaint.h"
#include "DisplayConstraints.h"
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
static constexpr int   N_STEPS = 320;
static constexpr int   KNOT_P  = 2;
static constexpr int   KNOT_Q  = 3;
static constexpr float TORUS_MAJOR = 0.62f;
static constexpr float TORUS_MINOR = 0.22f;
static constexpr float HEIGHT_GAIN = 0.95f;

// Keep the knot displaced from the axis dead-core with extra margin for thickness.
static constexpr float KNOT_EXTRA_Z_MARGIN_PX = 12.0f;
static constexpr float Z_BIAS_PX = CORE_SAFE_RADIUS_PX + KNOT_EXTRA_Z_MARGIN_PX;

static constexpr float SCALE_MIN_PX = 10.0f;
static constexpr float SCALE_MAX_PX = 26.0f;
static constexpr float SCALE_MIN = SCALE_MIN_PX / (VOXEL_H - 1);
static constexpr float SCALE_MAX = SCALE_MAX_PX / (VOXEL_H - 1);

static constexpr float SMOOTHING         = 0.15f;
static constexpr float ANGULAR_GAIN      = 0.12f;
static constexpr float ANGULAR_VEL_DECAY = 0.92f;

static const uint8_t KNOT_A_RGB[3] = {255, 60, 220};
static const uint8_t KNOT_B_RGB[3] = {0, 220, 255};

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void TorusKnotApp::setup(Renderer& /*renderer*/) {}

void TorusKnotApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    if (!hand.hand_detected) {
        angularVel_ *= ANGULAR_VEL_DECAY;
        phase_ += angularVel_;
        return;
    }

    Gesture g = detectGesture(hand);

    if (g == Gesture::PEACE) {
        float dirX = hand.lm_x[8] - hand.lm_x[5];
        dirX = clampf(dirX, -0.5f, 0.5f);
        float target = dirX * 2.0f * ANGULAR_GAIN;
        angularVel_ += (target - angularVel_) * SMOOTHING;
    } else {
        angularVel_ *= ANGULAR_VEL_DECAY;
    }

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

void TorusKnotApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    int prevX = 0, prevY = 0, prevZ = 0;
    bool havePrev = false;

    for (int i = 0; i <= N_STEPS; ++i) {
        float u = (float)i / (float)N_STEPS;
        float t = u * 2.0f * (float)M_PI;

        float cqt = cosf((float)KNOT_Q * t);
        float sqt = sinf((float)KNOT_Q * t);
        float cpt = cosf((float)KNOT_P * t);
        float spt = sinf((float)KNOT_P * t);

        float ring = TORUS_MAJOR + TORUS_MINOR * cqt;
        float x = ring * cpt;
        float y = HEIGHT_GAIN * TORUS_MINOR * sqt;
        float z = ring * spt;

        float cp = cosf(phase_);
        float sp = sinf(phase_);
        float xr =  cp * x - sp * z;
        float zr =  sp * x + cp * z;

        int vx = (int)roundf((xr * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1));
        int vy = (int)roundf((y  * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1));
        int vz = (int)roundf((zr * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1) + Z_BIAS_PX);
        vx = std::max(0, std::min(VOXEL_W - 1, vx));
        vy = std::max(0, std::min(VOXEL_H - 1, vy));
        vz = std::max(0, std::min(VOXEL_D - 1, vz));

        if (havePrev) {
            const uint8_t* c = (i & 1) ? KNOT_A_RGB : KNOT_B_RGB;
            voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz,
                                  c[0], c[1], c[2]);
        }

        prevX = vx; prevY = vy; prevZ = vz;
        havePrev = true;
    }

    renderer.uploadVoxelBuffer(voxels);
}
