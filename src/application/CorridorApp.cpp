#include "CorridorApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr float CAM_X_SCALE    = 80.0f;   // finger [0,1] -> world [-40, 40]
static constexpr float CAM_Z_SCALE    = 300.0f;  // finger [0,1] -> world [0, 300]
static constexpr float SMOOTHING      = 0.08f;

static constexpr float AMP            = 30.0f;   // sine wave amplitude (world units)
static constexpr float PERIOD         = 100.0f;  // sine wave Z period (world units)

static constexpr int   N_SLICES       = 40;      // number of depth samples
static constexpr float HALF_W         = 18.0f;   // corridor half-width
static constexpr int   WALL_H         = 40;      // wall height in voxels

// Wall color (blue-white)
static constexpr uint8_t WR = 120, WG = 180, WB = 255;
// Floor color (slightly dimmer)
static constexpr uint8_t FR = 70, FG = 110, FB = 200;

static inline float corridorCenterX(float wz)
{
    return AMP * sinf(wz / PERIOD);
}

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CorridorApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
    camX_ = 0.0f;
    camZ_ = 64.0f;
}

void CorridorApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);
    if (!hand.hand_detected) return;

    // Map index fingertip to camera world position
    float targetX = (hand.lm_x[8] - 0.5f) * CAM_X_SCALE;
    float targetZ =  hand.lm_y[8]          * CAM_Z_SCALE;

    camX_ += (targetX - camX_) * SMOOTHING;
    camZ_ += (targetZ - camZ_) * SMOOTHING;
}

void CorridorApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    const float worldZ_start = camZ_ - VOXEL_D * 0.5f;
    const float step         = (float)VOXEL_D / (float)N_SLICES;

    int prevVxL = 0, prevVxR = 0, prevVz = 0;
    bool havePrev = false;

    for (int i = 0; i <= N_SLICES; ++i) {
        float wz  = worldZ_start + i * step;
        float cx  = corridorCenterX(wz);
        float wxL = cx - HALF_W;
        float wxR = cx + HALF_W;

        int vz  = (int)(wz  - camZ_ + VOXEL_D * 0.5f + 0.5f);
        int vxL = (int)(wxL - camX_ + VOXEL_W * 0.5f + 0.5f);
        int vxR = (int)(wxR - camX_ + VOXEL_W * 0.5f + 0.5f);

        // Vertical wall columns at this slice
        voxpaint::paint3DLine(voxels, vxL, 0, vz, vxL, WALL_H, vz, WR, WG, WB);
        voxpaint::paint3DLine(voxels, vxR, 0, vz, vxR, WALL_H, vz, WR, WG, WB);

        if (havePrev) {
            // Left wall top edge
            voxpaint::paint3DLine(voxels, prevVxL, WALL_H, prevVz,
                                          vxL,     WALL_H, vz,    WR, WG, WB);
            // Right wall top edge
            voxpaint::paint3DLine(voxels, prevVxR, WALL_H, prevVz,
                                          vxR,     WALL_H, vz,    WR, WG, WB);
            // Left floor edge
            voxpaint::paint3DLine(voxels, prevVxL, 0, prevVz,
                                          vxL,     0, vz,         FR, FG, FB);
            // Right floor edge
            voxpaint::paint3DLine(voxels, prevVxR, 0, prevVz,
                                          vxR,     0, vz,         FR, FG, FB);
        }

        prevVxL  = vxL;
        prevVxR  = vxR;
        prevVz   = vz;
        havePrev = true;
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
