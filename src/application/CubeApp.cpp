#include "CubeApp.h"
#include "VoxelPaint.h"
#include "DisplayConstraints.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>

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
// Inline helpers
// -----------------------------------------------------------------------
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------
// Tunable demo variables
// -----------------------------------------------------------------------
static constexpr float SCALE_MIN_PX     = 8.0f;
static constexpr float SCALE_MAX_PX     = 24.0f;
static constexpr float SMOOTHING_FACTOR = 0.1f;

static constexpr float SCALE_MIN = SCALE_MIN_PX / (VOXEL_H - 1);
static constexpr float SCALE_MAX = SCALE_MAX_PX / (VOXEL_H - 1);

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CubeApp::setup(Renderer& /*renderer*/) {}

void CubeApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);
    if (!hand.hand_detected) {
        posX_ *= 0.98f;
        posZ_ += (0.0f - posZ_) * SMOOTHING_FACTOR;   // drift back to center
        return;
    }

    // MediaPipe joint indices: 0=wrist, 5=index_mcp, 9=middle_mcp,
    // 17=pinky_mcp, 4=thumb_tip, 8=index_tip
    float tgtRotX = (hand.lm_y[9]  - hand.lm_y[0]) * 2.0f * (float)M_PI;
    float tgtRotY = ((hand.lm_x[0] + hand.lm_x[9]) * 0.5f - 0.5f) * 2.0f * (float)M_PI;
    float tgtRotZ = atan2f(hand.lm_y[17] - hand.lm_y[5],
                           hand.lm_x[17] - hand.lm_x[5]);

    float pinch   = hypotf(hand.lm_x[4] - hand.lm_x[8],
                           hand.lm_y[4] - hand.lm_y[8]);
    float tgtScale = SCALE_MIN + clampf((pinch - 0.03f) / 0.22f, 0.0f, 1.0f)
                                 * (SCALE_MAX - SCALE_MIN);

    // Camera X → voxel X, camera Y → voxel Z (horizontal plane)
    float palmX   = (hand.lm_x[0] + hand.lm_x[9]) * 0.5f;
    float palmY   = (hand.lm_y[0] + hand.lm_y[9]) * 0.5f;
    float tgtPosX = clampf((palmX - 0.5f) * (float)VOXEL_W, -48.0f, 48.0f);
    float tgtPosZ = clampf((palmY - 0.5f) * (float)VOXEL_D, -48.0f, 48.0f);

    // Exponential smoothing
    rotX_  += (tgtRotX  - rotX_)  * SMOOTHING_FACTOR;
    rotY_  += (tgtRotY  - rotY_)  * SMOOTHING_FACTOR;
    rotZ_  += (tgtRotZ  - rotZ_)  * SMOOTHING_FACTOR;
    scale_ += (tgtScale - scale_) * SMOOTHING_FACTOR;
    scale_  = clampf(scale_, SCALE_MIN, SCALE_MAX);
    posX_  += (tgtPosX  - posX_)  * SMOOTHING_FACTOR;
    posZ_  += (tgtPosZ  - posZ_)  * SMOOTHING_FACTOR;
}

void CubeApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    const uint8_t R = 0, G = 255, B = 255;

    float transformed[8][3];
    for (int vi = 0; vi < 8; ++vi) {
        float rot[3];
        voxpaint::rotateXYZ(rotX_, rotY_, rotZ_, kVerts[vi], rot);

        transformed[vi][0] = (rot[0] * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1) + posX_;
        transformed[vi][1] = (rot[1] * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1) + posY_;
        transformed[vi][2] = (rot[2] * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1) + posZ_;
    }

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

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
