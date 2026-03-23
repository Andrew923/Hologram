#include "WireframeApp.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Gesture thresholds
// -----------------------------------------------------------------------
// Rotation gesture: index tip (8) and middle tip (12) must be within this
// normalized distance to activate rotation mode.
static constexpr float ROTATION_FINGER_PROXIMITY = 0.08f;

// Scale gesture: fingers other than thumb+index must have tip closer to
// wrist than MCP by this margin to count as "curled".
// (We check ring and pinky; middle is used for rotation detection.)
static constexpr float SCALE_SPIN_MAX = 0.06f; // max angular velocity from gesture

// Friction: auto-spin decays slowly when hand is absent
static constexpr float SPIN_FRICTION = 0.995f;

// Default auto-spin speed (rad/frame) when no gesture has been made yet
static constexpr float DEFAULT_SPIN_Y = 0.02f;

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------
WireframeApp::WireframeApp(const std::string& objPath)
    : objPath_(objPath)
{}

// -----------------------------------------------------------------------
// Setup: load OBJ and compute bounding box
// -----------------------------------------------------------------------
void WireframeApp::setup(Renderer& /*renderer*/)
{
    if (objPath_.empty()) {
        fprintf(stderr, "WireframeApp: no OBJ path provided\n");
        return;
    }

    if (!loadObj(objPath_, mesh_)) {
        fprintf(stderr, "WireframeApp: failed to load mesh\n");
        return;
    }

    computeBBox();
}

void WireframeApp::computeBBox()
{
    if (mesh_.vertices.empty()) return;

    bboxMin_[0] = bboxMax_[0] = mesh_.vertices[0][0];
    bboxMin_[1] = bboxMax_[1] = mesh_.vertices[0][1];
    bboxMin_[2] = bboxMax_[2] = mesh_.vertices[0][2];

    for (auto& v : mesh_.vertices) {
        for (int i = 0; i < 3; ++i) {
            bboxMin_[i] = std::min(bboxMin_[i], v[i]);
            bboxMax_[i] = std::max(bboxMax_[i], v[i]);
        }
    }

    float maxExtent = 0.0f;
    for (int i = 0; i < 3; ++i) {
        bboxCenter_[i] = (bboxMin_[i] + bboxMax_[i]) * 0.5f;
        maxExtent = std::max(maxExtent, bboxMax_[i] - bboxMin_[i]);
    }

    // Scale factor to normalize the model into [-1, 1] range
    bboxScale_ = (maxExtent > 1e-6f) ? (2.0f / maxExtent) : 1.0f;
}

// -----------------------------------------------------------------------
// Finger state helpers
// -----------------------------------------------------------------------
float WireframeApp::landmarkDist(const SharedHandData& hand, int a, int b)
{
    float dx = hand.lm_x[a] - hand.lm_x[b];
    float dy = hand.lm_y[a] - hand.lm_y[b];
    return sqrtf(dx * dx + dy * dy);
}

bool WireframeApp::isFingerExtended(const SharedHandData& hand, int tipIdx, int mcpIdx)
{
    // A finger is extended if its tip is farther from the wrist (0)
    // than its MCP joint is.
    float tipDist = landmarkDist(hand, tipIdx, 0);
    float mcpDist = landmarkDist(hand, mcpIdx, 0);
    return tipDist > mcpDist;
}

// -----------------------------------------------------------------------
// Update: process gestures
// -----------------------------------------------------------------------
void WireframeApp::update(const SharedHandData& hand)
{
    handPresent_ = hand.hand_detected;

    if (!hand.hand_detected) {
        rotationActive_ = false;
        scaleActive_ = false;

        // Auto-spin: apply current velocity with friction
        rotX_ += spinVelX_;
        rotY_ += spinVelY_;
        rotZ_ += spinVelZ_;
        spinVelX_ *= SPIN_FRICTION;
        spinVelY_ *= SPIN_FRICTION;
        spinVelZ_ *= SPIN_FRICTION;

        // If spin has nearly stopped, restore gentle default spin
        float totalSpin = fabsf(spinVelX_) + fabsf(spinVelY_) + fabsf(spinVelZ_);
        if (totalSpin < 0.001f) {
            spinVelY_ = DEFAULT_SPIN_Y;
        }
        return;
    }

    // --- Detect rotation gesture ---
    // Index tip (8) and middle tip (12) must be close together
    float indexMiddleDist = landmarkDist(hand, 8, 12);
    rotationActive_ = (indexMiddleDist < ROTATION_FINGER_PROXIMITY);

    // --- Detect scale gesture ---
    // All fingers except thumb (4) and index (8) must be curled.
    // Check middle (12/9), ring (16/13), pinky (20/17):
    bool middleCurled = !isFingerExtended(hand, 12, 9);
    bool ringCurled   = !isFingerExtended(hand, 16, 13);
    bool pinkyCurled  = !isFingerExtended(hand, 20, 17);
    bool thumbOut     = isFingerExtended(hand, 4, 2);
    bool indexOut     = isFingerExtended(hand, 8, 5);
    scaleActive_ = middleCurled && ringCurled && pinkyCurled && thumbOut && indexOut;

    // --- Apply rotation gesture ---
    if (rotationActive_) {
        // Midpoint of index+middle tips gives the "pointer" position
        float px = (hand.lm_x[8] + hand.lm_x[12]) * 0.5f;
        float py = (hand.lm_y[8] + hand.lm_y[12]) * 0.5f;

        // Offset from screen center [0.5, 0.5] determines angular velocity
        float dx = px - 0.5f;  // left/right → Y rotation
        float dy = py - 0.5f;  // up/down → X rotation

        // Scale to angular velocity (max ~0.06 rad/frame ≈ ~3.4°/frame)
        spinVelX_ = -dy * SCALE_SPIN_MAX * 2.0f;
        spinVelY_ =  dx * SCALE_SPIN_MAX * 2.0f;
        spinVelZ_ = 0.0f;

        rotX_ += spinVelX_;
        rotY_ += spinVelY_;
    }

    // --- Apply scale gesture ---
    if (scaleActive_) {
        float pinchDist = landmarkDist(hand, 4, 8);

        // Map pinch distance [0.03 .. 0.25] → scale [0.3 .. 2.0]
        float t = (pinchDist - 0.03f) / 0.22f;
        t = std::max(0.0f, std::min(1.0f, t));
        float targetScale = 0.3f + t * 1.7f;

        // Smooth towards target
        scale_ += (targetScale - scale_) * 0.3f;
    }

    // If neither gesture is active but hand is present, let it coast
    if (!rotationActive_ && !scaleActive_) {
        rotX_ += spinVelX_;
        rotY_ += spinVelY_;
        rotZ_ += spinVelZ_;
        spinVelX_ *= SPIN_FRICTION;
        spinVelY_ *= SPIN_FRICTION;
        spinVelZ_ *= SPIN_FRICTION;
    }
}

// -----------------------------------------------------------------------
// Rotation matrix (same as CubeApp)
// -----------------------------------------------------------------------
void WireframeApp::rotate(const float v[3], float out[3]) const
{
    float cx = cosf(rotX_), sx = sinf(rotX_);
    float cy = cosf(rotY_), sy = sinf(rotY_);
    float cz = cosf(rotZ_), sz = sinf(rotZ_);

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

    out[0] = v[0]*R[0][0] + v[1]*R[1][0] + v[2]*R[2][0];
    out[1] = v[0]*R[0][1] + v[1]*R[1][1] + v[2]*R[2][1];
    out[2] = v[0]*R[0][2] + v[1]*R[1][2] + v[2]*R[2][2];
}

// -----------------------------------------------------------------------
// Voxel painting (identical to CubeApp)
// -----------------------------------------------------------------------
void WireframeApp::paintVoxel(uint8_t* voxels, int x, int y, int z,
                               uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= VOXEL_W) return;
    if (y < 0 || y >= VOXEL_H) return;
    if (z < 0 || z >= VOXEL_D) return;
    int idx = ((z * VOXEL_H + y) * VOXEL_W + x) * 4;
    voxels[idx + 0] = r;
    voxels[idx + 1] = g;
    voxels[idx + 2] = b;
    voxels[idx + 3] = 255;
}

void WireframeApp::paint3DLine(uint8_t* voxels,
                                int x0, int y0, int z0,
                                int x1, int y1, int z1,
                                uint8_t r, uint8_t g, uint8_t b)
{
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int dz = std::abs(z1 - z0);
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
// Draw: transform mesh and paint into voxel buffer
// -----------------------------------------------------------------------
void WireframeApp::draw(Renderer& renderer)
{
    if (mesh_.vertices.empty() || mesh_.edges.empty()) return;

    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    const uint8_t R = 0, G = 255, B = 255; // cyan wireframe

    int numVerts = (int)mesh_.vertices.size();

    // Transform all vertices: center, normalize, rotate, scale, map to voxel space
    std::vector<float> tx(numVerts), ty(numVerts), tz(numVerts);
    for (int i = 0; i < numVerts; ++i) {
        // Center and normalize to [-1, 1]
        float v[3];
        v[0] = (mesh_.vertices[i][0] - bboxCenter_[0]) * bboxScale_;
        v[1] = (mesh_.vertices[i][1] - bboxCenter_[1]) * bboxScale_;
        v[2] = (mesh_.vertices[i][2] - bboxCenter_[2]) * bboxScale_;

        // Rotate
        float rot[3];
        rotate(v, rot);

        // Scale and map to voxel space
        tx[i] = (rot[0] * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1);
        ty[i] = (rot[1] * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1);
        tz[i] = (rot[2] * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1);
    }

    // Paint all edges
    for (auto& edge : mesh_.edges) {
        int a = edge[0], b = edge[1];
        if (a < 0 || a >= numVerts || b < 0 || b >= numVerts) continue;

        paint3DLine(voxels,
                    (int)roundf(tx[a]), (int)roundf(ty[a]), (int)roundf(tz[a]),
                    (int)roundf(tx[b]), (int)roundf(ty[b]), (int)roundf(tz[b]),
                    R, G, B);
    }

    renderer.uploadVoxelBuffer(voxels);
}
