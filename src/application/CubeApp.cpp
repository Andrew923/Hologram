#include "CubeApp.h"
#include "VoxelPaint.h"
#include "DisplayConstraints.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <cstdio>
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

// Depth mapping constants (mirrors ParticleApp)
static constexpr float Z_MIN_VOXEL = 16.0f, Z_MAX_VOXEL = 112.0f;
static constexpr float FALLBACK_BONE_PX_NEAR = 0.22f;
static constexpr float FALLBACK_BONE_PX_FAR  = 0.06f;

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CubeApp::setup(Renderer& /*renderer*/)
{
    camOk_ = cam_.loadFromFile("config/camera.json");
    if (camOk_) {
        fprintf(stderr,
                "CubeApp: loaded camera config (fx=%.1f bone=%.3fm)\n",
                cam_.fx, cam_.user_index_bone_m);
    } else {
        fprintf(stderr, "CubeApp: no camera.json, using hand-size depth proxy\n");
    }
}

void CubeApp::update(const SharedHandData& hand)
{
    static constexpr float CENTER_Z = 0.5f * (VOXEL_D - 1);   // 63.5 voxels

    if (!hand.hand_detected) {
        posX_ *= 0.98f;
        posY_ *= 0.98f;
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

    // XY position: palm center mapped to full voxel range
    float palmX   = (hand.lm_x[0] + hand.lm_x[9]) * 0.5f;
    float palmY   = (hand.lm_y[0] + hand.lm_y[9]) * 0.5f;
    float tgtPosX = clampf((palmX - 0.5f) * (float)VOXEL_W, -48.0f, 48.0f);
    float tgtPosY = clampf((palmY - 0.5f) * (float)VOXEL_H, -24.0f, 24.0f);

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
            zVoxel = Z_MAX_VOXEL - tf * (Z_MAX_VOXEL - Z_MIN_VOXEL);
        } else {
            float tf = clampf((bone_norm - FALLBACK_BONE_PX_FAR)
                              / (FALLBACK_BONE_PX_NEAR - FALLBACK_BONE_PX_FAR),
                              0.0f, 1.0f);
            zVoxel = Z_MIN_VOXEL + tf * (Z_MAX_VOXEL - Z_MIN_VOXEL);
        }
        smoothedZ_ += (zVoxel - smoothedZ_) * 0.25f;
    }
    // posZ_ is an offset from the model center (63.5); map absolute depth → offset.
    float tgtPosZ = smoothedZ_ - CENTER_Z;

    // Exponential smoothing
    rotX_  += (tgtRotX  - rotX_)  * SMOOTHING_FACTOR;
    rotY_  += (tgtRotY  - rotY_)  * SMOOTHING_FACTOR;
    rotZ_  += (tgtRotZ  - rotZ_)  * SMOOTHING_FACTOR;
    scale_ += (tgtScale - scale_) * SMOOTHING_FACTOR;
    scale_  = clampf(scale_, SCALE_MIN, SCALE_MAX);
    posX_  += (tgtPosX  - posX_)  * SMOOTHING_FACTOR;
    posY_  += (tgtPosY  - posY_)  * SMOOTHING_FACTOR;
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

    renderer.uploadVoxelBuffer(voxels);
}
