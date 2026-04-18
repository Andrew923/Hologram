#include "HandApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <algorithm>
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

// Depth mapping constants (mirrors ParticleApp)
static constexpr float Z_MIN = 16.0f, Z_MAX = 112.0f;
static constexpr float FALLBACK_BONE_PX_NEAR = 0.22f;
static constexpr float FALLBACK_BONE_PX_FAR  = 0.06f;

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void HandApp::setup(Renderer& /*renderer*/)
{
    camOk_ = cam_.loadFromFile("config/camera.json");
    if (camOk_) {
        fprintf(stderr,
                "HandApp: loaded camera config (fx=%.1f bone=%.3fm)\n",
                cam_.fx, cam_.user_index_bone_m);
    } else {
        fprintf(stderr, "HandApp: no camera.json, using hand-size depth proxy\n");
    }

    for (int i = 0; i < 21; ++i) {
        filtersX_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
        filtersY_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
    }
}

void HandApp::update(const SharedHandData& hand)
{
    anyValid_ = false;
    if (!hand.hand_detected) {
        posX_ += (64.0f - posX_) * 0.02f;
        posY_ += (32.0f - posY_) * 0.02f;

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

    for (int i = 0; i < 21; ++i) {
        float rawX = hand.lm_x[i] * 128.0f;
        float rawY = hand.lm_y[i] * 64.0f;
        smoothX_[i] = filtersX_[i].filter(rawX, t);
        smoothY_[i] = filtersY_[i].filter(rawY, t);
    }

    // Palm anchor clamping
    float palmX   = (hand.lm_x[0] + hand.lm_x[9]) * 0.5f;
    float palmY   = (hand.lm_y[0] + hand.lm_y[9]) * 0.5f;
    float tgtPosX = clampf(palmX * 128.0f, 16.0f, 112.0f);
    float tgtPosY = clampf(palmY *  64.0f,  8.0f,  56.0f);
    posX_ += (tgtPosX - posX_) * 0.3f;
    posY_ += (tgtPosY - posY_) * 0.3f;

    // Depth estimation: wrist(0) → index MCP(5) bone length
    float dx_n = hand.lm_x[5] - hand.lm_x[0];
    float dy_n = hand.lm_y[5] - hand.lm_y[0];
    float bone_norm = hypotf(dx_n, dy_n);
    if (bone_norm >= 1e-4f) {
        float zVoxel;
        if (camOk_) {
            float L_px = std::max(hypotf(
                dx_n * (float)cam_.image_width,
                dy_n * (float)cam_.image_height), 1.0f);
            float Z_m = cam_.fx * cam_.user_index_bone_m / L_px;
            float tf = clampf((Z_m - 0.25f) / 0.50f, 0.0f, 1.0f);
            zVoxel = Z_MAX - tf * (Z_MAX - Z_MIN);
        } else {
            float tf = clampf((bone_norm - FALLBACK_BONE_PX_FAR)
                              / (FALLBACK_BONE_PX_NEAR - FALLBACK_BONE_PX_FAR),
                              0.0f, 1.0f);
            zVoxel = Z_MIN + tf * (Z_MAX - Z_MIN);
        }
        smoothedZ_ += (zVoxel - smoothedZ_) * 0.25f;
    }

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
        float palCY = (smoothY_[0] + smoothY_[9]) * 0.5f;
        float offX  = posX_ - palCX;
        float offY  = posY_ - palCY;
        int iz = (int)roundf(clampf(smoothedZ_, Z_MIN, Z_MAX));

        // Draw 3D bones
        for (auto& conn : CONNECTIONS) {
            int j1 = conn[0], j2 = conn[1];
            int x1 = (int)(smoothX_[j1] + offX), y1 = (int)(smoothY_[j1] + offY);
            int x2 = (int)(smoothX_[j2] + offX), y2 = (int)(smoothY_[j2] + offY);
            if ((x1 == 0 && y1 == 0) || (x2 == 0 && y2 == 0)) continue;
            voxpaint::paint3DLine(voxels, x1, y1, iz, x2, y2, iz, 255, 255, 255);
        }

        // Draw joints
        for (int i = 0; i < 21; ++i) {
            int ix = (int)(smoothX_[i] + offX);
            int iy = (int)(smoothY_[i] + offY);
            if (ix == 0 && iy == 0) continue;
            if (isFingertip(i))
                voxpaint::paintCube(voxels, ix, iy, iz, 1, 255, 0, 0);
            else
                voxpaint::paintVoxel(voxels, ix, iy, iz, 0, 255, 0);
        }
    }

    renderer.uploadVoxelBuffer(voxels);
}
