#include "HandApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <initializer_list>

// constexpr definitions (required in some C++17 contexts)
constexpr int HandApp::CONNECTIONS[20][2];
constexpr int HandApp::FINGERTIP_INDICES[5];

static double nowSeconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool isFingertip(int idx) {
    for (int t : {4, 8, 12, 16, 20})
        if (idx == t) return true;
    return false;
}

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Fixed vertical midpoint in the 64-high voxel grid
static constexpr int IY = 32;

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void HandApp::setup(Renderer& /*renderer*/)
{
    for (int i = 0; i < 21; ++i) {
        filtersX_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
        filtersZ_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
    }
}

void HandApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);
    anyValid_ = false;
    if (!hand.hand_detected) {
        posX_ += (64.0f - posX_) * 0.02f;
        posZ_ += (64.0f - posZ_) * 0.02f;

        if (lastGesture_ != Gesture::NONE) {
            printf("[gesture] NONE\n");
            fflush(stdout);
            lastGesture_ = Gesture::NONE;
        }
        return;
    }

    if (hand.lm_x[0] == 0.0f && hand.lm_y[0] == 0.0f) return;

    anyValid_ = true;
    double t = hand.timestamp > 0.0 ? hand.timestamp : nowSeconds();

    // Camera X → voxel X, camera Y → voxel Z (horizontal plane)
    for (int i = 0; i < 21; ++i) {
        float rawX = hand.lm_x[i] * 128.0f;
        float rawZ = hand.lm_y[i] * 128.0f;
        smoothX_[i] = filtersX_[i].filter(rawX, t);
        smoothZ_[i] = filtersZ_[i].filter(rawZ, t);
    }

    // Palm anchor clamping in the horizontal plane
    float palmX   = (hand.lm_x[0] + hand.lm_x[9]) * 0.5f;
    float palmY   = (hand.lm_y[0] + hand.lm_y[9]) * 0.5f;
    float tgtPosX = clampf(palmX * 128.0f, 16.0f, 112.0f);
    float tgtPosZ = clampf(palmY * 128.0f, 16.0f, 112.0f);
    posX_ += (tgtPosX - posX_) * 0.3f;
    posZ_ += (tgtPosZ - posZ_) * 0.3f;

    // Gesture detection
    Gesture g = detectGesture(hand);
    if (g != lastGesture_) {
        printf("[gesture] %s\n", gestureName(g));
        fflush(stdout);
        lastGesture_ = g;
    }
}

void HandApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    if (anyValid_) {
        float palCX = (smoothX_[0] + smoothX_[9]) * 0.5f;
        float palCZ = (smoothZ_[0] + smoothZ_[9]) * 0.5f;
        float offX  = posX_ - palCX;
        float offZ  = posZ_ - palCZ;

        // Draw 3D bones flat on the horizontal plane (constant Y = IY)
        for (auto& conn : CONNECTIONS) {
            int j1 = conn[0], j2 = conn[1];
            int x1 = (int)(smoothX_[j1] + offX), z1 = (int)(smoothZ_[j1] + offZ);
            int x2 = (int)(smoothX_[j2] + offX), z2 = (int)(smoothZ_[j2] + offZ);
            if ((x1 == 0 && z1 == 0) || (x2 == 0 && z2 == 0)) continue;
            voxpaint::paint3DLine(voxels, x1, IY, z1, x2, IY, z2, 255, 255, 255);
        }

        // Draw joints
        for (int i = 0; i < 21; ++i) {
            int ix = (int)(smoothX_[i] + offX);
            int iz = (int)(smoothZ_[i] + offZ);
            if (ix == 0 && iz == 0) continue;
            if (isFingertip(i))
                voxpaint::paintCube(voxels, ix, IY, iz, 1, 255, 0, 0);
            else
                voxpaint::paintVoxel(voxels, ix, IY, iz, 0, 255, 0);
        }
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
