#include "CubeApp.h"
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
// Rotation helpers
// -----------------------------------------------------------------------
// Apply combined rotation R = Rz * Ry * Rx to vector v, store in out.
// Matches np.dot(v, get_rotation_matrix(rx, ry, rz)) from cube.py.
void CubeApp::rotate(const float v[3], float out[3]) const
{
    float cx = cosf(rotX_), sx = sinf(rotX_);
    float cy = cosf(rotY_), sy = sinf(rotY_);
    float cz = cosf(rotZ_), sz = sinf(rotZ_);

    // R = Rz * Ry * Rx  (3x3 matrix, row-major)
    // Row i of R combined = multiply out Rz*(Ry*Rx)
    // Following cube.py conventions: np.dot(Rz, np.dot(Ry, Rx))
    float R[3][3];
    R[0][0] =  cy*cz;
    R[0][1] =  cz*sx*sy - cx*sz;
    R[0][2] =  cx*cz*sy + sx*sz;
    R[1][0] =  cy*sz;
    R[1][1] =  cx*cz + sx*sy*sz;
    R[1][2] =  cx*sy*sz - cz*sx;
    R[2][0] = -sy;
    R[2][1] =  cy*sx;
    R[2][2] =  cx*cy;

    // np.dot(v, R) = v transposed times R = R^T * v (standard row-vec * matrix)
    out[0] = v[0]*R[0][0] + v[1]*R[1][0] + v[2]*R[2][0];
    out[1] = v[0]*R[0][1] + v[1]*R[1][1] + v[2]*R[2][1];
    out[2] = v[0]*R[0][2] + v[1]*R[1][2] + v[2]*R[2][2];
}

// -----------------------------------------------------------------------
// Voxel buffer helpers
// -----------------------------------------------------------------------
void CubeApp::paintVoxel(uint8_t* voxels, int x, int y, int z,
                          uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= VOXEL_W) return;
    if (y < 0 || y >= VOXEL_H) return;
    if (z < 0 || z >= VOXEL_D) return;
    // Layout: data[(z*VOXEL_H + y)*VOXEL_W + x]  (each element 4 bytes RGBA)
    int idx = ((z * VOXEL_H + y) * VOXEL_W + x) * 4;
    voxels[idx + 0] = r;
    voxels[idx + 1] = g;
    voxels[idx + 2] = b;
    voxels[idx + 3] = 255;
}

// 3D DDA line painter — interpolates along the dominant axis
void CubeApp::paint3DLine(uint8_t* voxels,
                           int x0, int y0, int z0,
                           int x1, int y1, int z1,
                           uint8_t r, uint8_t g, uint8_t b)
{
    int dx = std::abs(x1-x0);
    int dy = std::abs(y1-y0);
    int dz = std::abs(z1-z0);
    int dominant = std::max({dx, dy, dz});

    if (dominant == 0) {
        paintVoxel(voxels, x0, y0, z0, r, g, b);
        return;
    }

    for (int i = 0; i <= dominant; ++i) {
        float t = (float)i / (float)dominant;
        int x = (int)roundf(x0 + t * (x1 - x0));
        int y = (int)roundf(y0 + t * (y1 - y0));
        int z = (int)roundf(z0 + t * (z1 - z0));
        paintVoxel(voxels, x, y, z, r, g, b);
    }
}

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CubeApp::setup(Renderer& /*renderer*/) {}

void CubeApp::update(const SharedHandData& hand)
{
    if (!hand.hand_detected) return;

    // MediaPipe joint indices (native order):
    // 0=wrist, 5=index_mcp, 9=middle_mcp, 17=pinky_mcp, 4=thumb_tip, 8=index_tip
    // Rotation formulas from cube.py lines 289–332, re-indexed to MediaPipe
    float tgtRotX = (hand.lm_y[9]  - hand.lm_y[0]) * 2.0f * (float)M_PI;
    float tgtRotY = ((hand.lm_x[0] + hand.lm_x[9]) * 0.5f - 0.5f) * 2.0f * (float)M_PI;
    float tgtRotZ = atan2f(hand.lm_y[17] - hand.lm_y[5],
                           hand.lm_x[17] - hand.lm_x[5]);

    float pinch   = hypotf(hand.lm_x[4] - hand.lm_x[8],
                           hand.lm_y[4] - hand.lm_y[8]);
    float tgtScale = 0.3f + clampf((pinch - 0.03f) / 0.22f, 0.0f, 1.0f) * 1.7f;

    // Exponential smoothing — factor 0.3 matches cube.py smoothing_factor
    rotX_  += (tgtRotX  - rotX_)  * 0.3f;
    rotY_  += (tgtRotY  - rotY_)  * 0.3f;
    rotZ_  += (tgtRotZ  - rotZ_)  * 0.3f;
    scale_ += (tgtScale - scale_) * 0.3f;
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
        rotate(kVerts[vi], rot);

        // Scale and map to voxel space:
        //   Model x ∈ [-1,1] * scale_ → voxel X ∈ [0, VOXEL_W-1]  (center at W/2)
        //   Model y ∈ [-1,1] * scale_ → voxel Y ∈ [0, VOXEL_H-1]  (center at H/2)
        //   Model z ∈ [-1,1] * scale_ → voxel Z ∈ [0, VOXEL_D-1]  (center at D/2)
        transformed[vi][0] = (rot[0] * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1);
        transformed[vi][1] = (rot[1] * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1);
        transformed[vi][2] = (rot[2] * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1);
    }

    // Paint all 12 edges using 3D DDA
    for (int ei = 0; ei < 12; ++ei) {
        int a = kEdges[ei][0];
        int b = kEdges[ei][1];
        paint3DLine(voxels,
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
