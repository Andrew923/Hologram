#include "MorphApp.h"
#include "VoxelPaint.h"
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
static constexpr float SPIN_MAX      = 0.06f;   // max spin velocity (rad/frame)
static constexpr float SPIN_FRICTION = 0.995f;
static constexpr float DEFAULT_SPIN  = 0.02f;
static constexpr float MORPH_SMOOTH  = 0.04f;   // morphT smoothing rate
static constexpr float SCALE_PX      = 22.0f;   // half-extent in voxel space

static constexpr float PINCH_MIN  = 0.03f;
static constexpr float PINCH_SPAN = 0.22f;

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------
// Geometry: 4 polyhedra on the unit sphere, all sharing a common frame.
//
// Tetrahedron: vertices are alternating cube corners of the dodecahedron.
// Octahedron:  axis-aligned ±X/Y/Z.
// Icosahedron: dual of dodecahedron (face-centre directions).
// Dodecahedron: the full 20-slot canonical positions.
// -----------------------------------------------------------------------

static const float kTetraVerts[4][3] = {
    { 0.57735f,  0.57735f,  0.57735f},
    { 0.57735f, -0.57735f, -0.57735f},
    {-0.57735f,  0.57735f, -0.57735f},
    {-0.57735f, -0.57735f,  0.57735f},
};

static const float kOctaVerts[6][3] = {
    { 1.0f,  0.0f,  0.0f}, {-1.0f,  0.0f,  0.0f},
    { 0.0f,  1.0f,  0.0f}, { 0.0f, -1.0f,  0.0f},
    { 0.0f,  0.0f,  1.0f}, { 0.0f,  0.0f, -1.0f},
};

static const float kIcoVerts[12][3] = {
    { 0.85065f,  0.52573f,  0.00000f}, { 0.52573f,  0.00000f,  0.85065f},
    { 0.00000f,  0.85065f,  0.52573f}, { 0.52573f,  0.00000f, -0.85065f},
    { 0.00000f,  0.85065f, -0.52573f}, { 0.85065f, -0.52573f,  0.00000f},
    { 0.00000f, -0.85065f,  0.52573f}, { 0.00000f, -0.85065f, -0.52573f},
    {-0.85065f,  0.52573f,  0.00000f}, {-0.52573f,  0.00000f,  0.85065f},
    {-0.52573f,  0.00000f, -0.85065f}, {-0.85065f, -0.52573f,  0.00000f},
};

static const float kDodVerts[20][3] = {
    { 0.57735f,  0.57735f,  0.57735f}, { 0.57735f,  0.57735f, -0.57735f},
    { 0.57735f, -0.57735f,  0.57735f}, { 0.57735f, -0.57735f, -0.57735f},
    {-0.57735f,  0.57735f,  0.57735f}, {-0.57735f,  0.57735f, -0.57735f},
    {-0.57735f, -0.57735f,  0.57735f}, {-0.57735f, -0.57735f, -0.57735f},
    { 0.00000f,  0.35682f,  0.93417f}, { 0.00000f,  0.35682f, -0.93417f},
    { 0.00000f, -0.35682f,  0.93417f}, { 0.00000f, -0.35682f, -0.93417f},
    { 0.35682f,  0.93417f,  0.00000f}, { 0.35682f, -0.93417f,  0.00000f},
    {-0.35682f,  0.93417f,  0.00000f}, {-0.35682f, -0.93417f,  0.00000f},
    { 0.93417f,  0.00000f,  0.35682f}, { 0.93417f,  0.00000f, -0.35682f},
    {-0.93417f,  0.00000f,  0.35682f}, {-0.93417f,  0.00000f, -0.35682f},
};

// Slot → vertex index for each shape (20 slots total)
static const int kSlotToTetra[20] = {
    0, 0, 0, 1, 3, 2, 3, 2, 0, 2, 3, 1, 0, 1, 2, 3, 0, 1, 3, 2
};
static const int kSlotToOcta[20] = {
    0, 0, 0, 0, 1, 1, 1, 1, 4, 5, 4, 5, 2, 3, 2, 3, 0, 0, 1, 1
};
static const int kSlotToIco[20] = {
    0, 0, 5, 5, 8, 8,11,11, 1, 3, 9,10, 2, 6, 4, 7, 1, 3, 9,10
};
// Dodecahedron: slot s → vertex s (trivial)

// Representative slot per vertex for each shape
static const int kTetraRepSlot[4]  = { 0, 3, 5, 4 };
static const int kOctaRepSlot[6]   = { 0, 4, 12, 13, 8, 9 };
static const int kIcoRepSlot[12]   = { 0, 8, 12, 9, 14, 2, 13, 15, 4, 10, 11, 6 };
// Dodecahedron: repSlot[v] = v

// Edge lists expressed as representative slot index pairs
static const int kTetraEdgesSlots[6][2] = {
    {0,3},{0,5},{0,4},{3,5},{3,4},{5,4}
};
static const int kOctaEdgesSlots[12][2] = {
    {0,12},{0,13},{0,8},{0,9},
    {4,12},{4,13},{4,8},{4,9},
    {12,8},{12,9},{13,8},{13,9}
};
static const int kIcoEdgesSlots[30][2] = {
    {0,8},{0,12},{0,9},{0,14},{0,2},
    {8,12},{8,2},{8,13},{8,10},
    {12,14},{12,4},{12,10},
    {9,14},{9,2},{9,15},{9,11},
    {14,4},{14,11},
    {2,13},{2,15},
    {13,15},{13,10},{13,6},
    {15,11},{15,6},
    {4,10},{4,11},{4,6},
    {10,6},{11,6},
};
static const int kDodEdgesSlots[30][2] = {
    { 0, 8},{ 0,12},{ 0,16}, { 1, 9},{ 1,12},{ 1,17},
    { 2,10},{ 2,13},{ 2,16}, { 3,11},{ 3,13},{ 3,17},
    { 4, 8},{ 4,14},{ 4,18}, { 5, 9},{ 5,14},{ 5,19},
    { 6,10},{ 6,15},{ 6,18}, { 7,11},{ 7,15},{ 7,19},
    { 8,10},{ 9,11},{12,14},{13,15},{16,17},{18,19},
};

// Per-shape colors: tetra→octa→ico→dod
static const float kShapeColors[4][3] = {
    {  0.0f, 200.0f, 255.0f},  // cyan-blue
    {  0.0f, 255.0f, 100.0f},  // green
    {255.0f, 180.0f,   0.0f},  // amber
    {255.0f,  60.0f, 200.0f},  // magenta
};

// -----------------------------------------------------------------------
// SLERP for unit vectors
// -----------------------------------------------------------------------
static void slerp3(const float a[3], const float b[3], float t, float out[3])
{
    float cosTheta = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    cosTheta = clampf(cosTheta, -1.0f, 1.0f);
    if (cosTheta > 0.9999f) {
        float len = 0.0f;
        for (int i = 0; i < 3; ++i) { out[i] = a[i] + t*(b[i]-a[i]); len += out[i]*out[i]; }
        len = sqrtf(len);
        if (len > 1e-6f) { out[0]/=len; out[1]/=len; out[2]/=len; }
        return;
    }
    float theta  = acosf(cosTheta);
    float sinTh  = sinf(theta);
    float wa     = sinf((1.0f - t) * theta) / sinTh;
    float wb     = sinf(t * theta) / sinTh;
    out[0] = wa*a[0] + wb*b[0];
    out[1] = wa*a[1] + wb*b[1];
    out[2] = wa*a[2] + wb*b[2];
}

// Return the unit-sphere position of slot s for the given shape index (0-3)
static void slotPos(int shape, int slot, float out[3])
{
    switch (shape) {
        case 0: { int v = kSlotToTetra[slot]; out[0]=kTetraVerts[v][0]; out[1]=kTetraVerts[v][1]; out[2]=kTetraVerts[v][2]; break; }
        case 1: { int v = kSlotToOcta[slot];  out[0]=kOctaVerts[v][0];  out[1]=kOctaVerts[v][1];  out[2]=kOctaVerts[v][2];  break; }
        case 2: { int v = kSlotToIco[slot];   out[0]=kIcoVerts[v][0];   out[1]=kIcoVerts[v][1];   out[2]=kIcoVerts[v][2];   break; }
        default:{ out[0]=kDodVerts[slot][0];  out[1]=kDodVerts[slot][1]; out[2]=kDodVerts[slot][2]; break; }
    }
}

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void MorphApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
}

void MorphApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    if (!hand.hand_detected) {
        rotX_ += spinVelX_;
        rotY_ += spinVelY_;
        spinVelX_ *= SPIN_FRICTION;
        spinVelY_ *= SPIN_FRICTION;
        if (fabsf(spinVelX_) + fabsf(spinVelY_) < 0.001f)
            spinVelY_ = DEFAULT_SPIN;
        return;
    }

    // Index tip offset → rotation velocity
    float dx = hand.lm_x[8] - 0.5f;
    float dy = hand.lm_y[8] - 0.5f;
    spinVelX_ = -dy * SPIN_MAX * 2.0f;
    spinVelY_ =  dx * SPIN_MAX * 2.0f;
    rotX_ += spinVelX_;
    rotY_ += spinVelY_;

    // Pinch distance → target morphT [0,1]
    float pinchDist = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
    float targetMorphT = clampf((pinchDist - PINCH_MIN) / PINCH_SPAN, 0.0f, 1.0f);
    morphT_ += (targetMorphT - morphT_) * MORPH_SMOOTH;
}

void MorphApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    // Map morphT [0,1] → continuous position across 4 shapes
    float scaled  = morphT_ * 3.0f;
    int   shapeA  = (int)scaled;
    if (shapeA > 2) shapeA = 2;
    int   shapeB  = shapeA + 1;
    float frac    = scaled - (float)shapeA;

    // Interpolate color
    float cr = kShapeColors[shapeA][0] + frac * (kShapeColors[shapeB][0] - kShapeColors[shapeA][0]);
    float cg = kShapeColors[shapeA][1] + frac * (kShapeColors[shapeB][1] - kShapeColors[shapeA][1]);
    float cb = kShapeColors[shapeA][2] + frac * (kShapeColors[shapeB][2] - kShapeColors[shapeA][2]);
    uint8_t R = (uint8_t)clampf(cr, 0.0f, 255.0f);
    uint8_t G = (uint8_t)clampf(cg, 0.0f, 255.0f);
    uint8_t B = (uint8_t)clampf(cb, 0.0f, 255.0f);

    // Compute morphed positions for all 20 slots via SLERP
    float slotWorld[20][3];
    for (int s = 0; s < 20; ++s) {
        float posA[3], posB[3];
        slotPos(shapeA, s, posA);
        slotPos(shapeB, s, posB);
        slerp3(posA, posB, frac, slotWorld[s]);
    }

    // Rotate each slot position and project to voxel space
    float cx = 0.5f * (VOXEL_W - 1);
    float cy = 0.5f * (VOXEL_H - 1);
    float cz = 0.5f * (VOXEL_D - 1);

    float cosX = cosf(rotX_), sinX = sinf(rotX_);
    float cosY = cosf(rotY_), sinY = sinf(rotY_);

    float vx[20], vy[20], vz[20];
    for (int s = 0; s < 20; ++s) {
        float x = slotWorld[s][0];
        float y = slotWorld[s][1];
        float z = slotWorld[s][2];

        // Rotate Y then X
        float tx =  cosY * x + sinY * z;
        float tz = -sinY * x + cosY * z;
        x = tx; z = tz;
        float ty =  cosX * y - sinX * z;
        tz       =  sinX * y + cosX * z;
        y = ty; z = tz;

        vx[s] = cx + x * SCALE_PX;
        vy[s] = cy + y * SCALE_PX;
        vz[s] = cz + z * SCALE_PX;
    }

    // Draw edges: show shapeA edges when frac < 0.5, shapeB when frac >= 0.5
    const int (*edges)[2];
    int       nEdges;
    int       drawShape = (frac < 0.5f) ? shapeA : shapeB;

    switch (drawShape) {
        case 0: edges = kTetraEdgesSlots; nEdges = 6;  break;
        case 1: edges = kOctaEdgesSlots;  nEdges = 12; break;
        case 2: edges = kIcoEdgesSlots;   nEdges = 30; break;
        default:edges = kDodEdgesSlots;   nEdges = 30; break;
    }

    for (int i = 0; i < nEdges; ++i) {
        int sa = edges[i][0], sb = edges[i][1];
        voxpaint::paint3DLine(voxels,
            (int)roundf(vx[sa]), (int)roundf(vy[sa]), (int)roundf(vz[sa]),
            (int)roundf(vx[sb]), (int)roundf(vy[sb]), (int)roundf(vz[sb]),
            R, G, B);
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
