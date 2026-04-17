#include "CubeApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Cube geometry (from cube.py lines 186–191)
// -----------------------------------------------------------------------
static const float kVerts[8][3] = {
    {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
    {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
};
static const int kEdges[12][2] = {
    {0,1},{1,2},{2,3},{3,0},   // front face
    {4,5},{5,6},{6,7},{7,4},   // back face
    {0,4},{1,5},{2,6},{3,7}    // connecting edges
};

// -----------------------------------------------------------------------
// Inline clamp helper
// -----------------------------------------------------------------------
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------
// Tunable demo variables
// -----------------------------------------------------------------------
static constexpr float SCALE_MIN_PX     = 8.0f;   // minimum cube height in pixels
static constexpr float SCALE_MAX_PX     = 24.0f;  // maximum cube height in pixels
static constexpr float SMOOTHING_FACTOR = 0.1f;   // exponential smoothing for rot/scale/pos (0=frozen, 1=instant)
static constexpr float CORE_RADIUS_PX   = 12.0f;  // unswept radius from panel geometry
static constexpr float CORE_MARGIN_PX   = 2.0f;   // keep some visual margin from core

// Derived scale limits (cube height = scale * (VOXEL_H - 1) pixels)
static constexpr float SCALE_MIN = SCALE_MIN_PX / (VOXEL_H - 1);
static constexpr float SCALE_MAX = SCALE_MAX_PX / (VOXEL_H - 1);

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CubeApp::setup(Renderer& /*renderer*/) {}

void CubeApp::update(const SharedHandData& hand)
{
    auto updatePosZForCore = [&]() {
        float halfExtentZ = scale_ * 0.5f * (VOXEL_D - 1);
        float tgtPosZ = CORE_RADIUS_PX + CORE_MARGIN_PX + halfExtentZ;
        float maxPosZ = 0.5f * (VOXEL_D - 1) - halfExtentZ - 1.0f;
        tgtPosZ = clampf(tgtPosZ, 0.0f, std::max(0.0f, maxPosZ));
        posZ_ += (tgtPosZ - posZ_) * SMOOTHING_FACTOR;
    };

    if (!hand.hand_detected) {
        posX_ *= 0.98f;   // slow exponential drift back to center (~3s at 60fps)
        posY_ *= 0.98f;
        updatePosZForCore();
        return;
    }

    // MediaPipe joint indices (native order):
    // 0=wrist, 5=index_mcp, 9=middle_mcp, 17=pinky_mcp, 4=thumb_tip, 8=index_tip
    // Rotation formulas from cube.py lines 289–332, re-indexed to MediaPipe
    float tgtRotX = (hand.lm_y[9]  - hand.lm_y[0]) * 2.0f * (float)M_PI;
    float tgtRotY = ((hand.lm_x[0] + hand.lm_x[9]) * 0.5f - 0.5f) * 2.0f * (float)M_PI;
    float tgtRotZ = atan2f(hand.lm_y[17] - hand.lm_y[5],
                           hand.lm_x[17] - hand.lm_x[5]);

    float pinch   = hypotf(hand.lm_x[4] - hand.lm_x[8],
                           hand.lm_y[4] - hand.lm_y[8]);
    float tgtScale = SCALE_MIN + clampf((pinch - 0.03f) / 0.22f, 0.0f, 1.0f)
                                 * (SCALE_MAX - SCALE_MIN);

    // Position offset: palm center clamped to ±8 voxels (= ±4px on 64x32 display)
    float palmX   = (hand.lm_x[0] + hand.lm_x[9]) * 0.5f;
    float palmY   = (hand.lm_y[0] + hand.lm_y[9]) * 0.5f;
    float tgtPosX = clampf((palmX - 0.5f) * (float)VOXEL_W, -8.0f, 8.0f);
    float tgtPosY = clampf((palmY - 0.5f) * (float)VOXEL_H, -8.0f, 8.0f);

    // Exponential smoothing — controlled by SMOOTHING_FACTOR above
    rotX_  += (tgtRotX  - rotX_)  * SMOOTHING_FACTOR;
    rotY_  += (tgtRotY  - rotY_)  * SMOOTHING_FACTOR;
    rotZ_  += (tgtRotZ  - rotZ_)  * SMOOTHING_FACTOR;
    scale_ += (tgtScale - scale_) * SMOOTHING_FACTOR;
    scale_  = clampf(scale_, SCALE_MIN, SCALE_MAX);
    posX_  += (tgtPosX  - posX_)  * SMOOTHING_FACTOR;
    posY_  += (tgtPosY  - posY_)  * SMOOTHING_FACTOR;
    updatePosZForCore();
}

void CubeApp::draw(Renderer& renderer)
{
    // Allocate CPU voxel buffer (4MB, zero-initialized)
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    // Bright cyan wireframe color
    const uint8_t R = 0, G = 255, B = 255;

    // Transform all 8 vertices: rotate then scale, then map to voxel space
    float transformed[8][3];
    for (int vi = 0; vi < 8; ++vi) {
        float rot[3];
        voxpaint::rotateXYZ(rotX_, rotY_, rotZ_, kVerts[vi], rot);

        // Scale and map to voxel space:
        //   Model x ∈ [-1,1] * scale_ → voxel X ∈ [0, VOXEL_W-1]  (center at W/2)
        //   Model y ∈ [-1,1] * scale_ → voxel Y ∈ [0, VOXEL_H-1]  (center at H/2)
        //   Model z ∈ [-1,1] * scale_ → voxel Z ∈ [0, VOXEL_D-1]  (center at D/2)
        transformed[vi][0] = (rot[0] * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1) + posX_;
        transformed[vi][1] = (rot[1] * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1) + posY_;
        transformed[vi][2] = (rot[2] * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1) + posZ_;
    }

    // Paint all 12 edges using 3D DDA
    for (int ei = 0; ei < 12; ++ei) {
        int a = kEdges[ei][0];
        int b = kEdges[ei][1];
        voxpaint::paint3DLine(voxels,
                    (int)roundf(transformed[a][0]),
                    (int)roundf(transformed[a][1]),
                    (int)roundf(transformed[a][2]),
                    (int)roundf(transformed[b][0]),
                    (int)roundf(transformed[b][1]),
                    (int)roundf(transformed[b][2]),
                    R, G, B);
    }

    renderer.uploadVoxelBuffer(voxels);
}
