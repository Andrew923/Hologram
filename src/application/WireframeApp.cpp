#include "WireframeApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Interaction constants
// -----------------------------------------------------------------------
static constexpr float SCALE_SPIN_MAX = 0.06f; // max angular velocity (rad/frame)

// Friction: auto-spin decays slowly when hand is absent
static constexpr float SPIN_FRICTION = 0.995f;

// Default auto-spin speed (rad/frame) when no gesture has been made yet
static constexpr float DEFAULT_SPIN_Y = 0.02f;

// -----------------------------------------------------------------------
// Model loading
// -----------------------------------------------------------------------
bool WireframeApp::setModel(const std::string& objPath)
{
    objPath_ = objPath;
    mesh_.vertices.clear();
    mesh_.edges.clear();

    if (objPath_.empty()) {
        fprintf(stderr, "WireframeApp: empty OBJ path\n");
        return false;
    }

    if (!loadObj(objPath_, mesh_)) {
        fprintf(stderr, "WireframeApp: failed to load mesh '%s'\n",
                objPath_.c_str());
        return false;
    }

    computeBBox();
    return true;
}

// -----------------------------------------------------------------------
// Setup: if a model path was set prior to setup, load it now.
// -----------------------------------------------------------------------
void WireframeApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
    if (!objPath_.empty() && mesh_.vertices.empty()) {
        if (!loadObj(objPath_, mesh_)) {
            fprintf(stderr, "WireframeApp: failed to load mesh '%s'\n",
                    objPath_.c_str());
            return;
        }
        computeBBox();
    }
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
// Helpers
// -----------------------------------------------------------------------
float WireframeApp::landmarkDist(const SharedHandData& hand, int a, int b)
{
    float dx = hand.lm_x[a] - hand.lm_x[b];
    float dy = hand.lm_y[a] - hand.lm_y[b];
    return sqrtf(dx * dx + dy * dy);
}

// -----------------------------------------------------------------------
// Update: index finger position → rotation, pinch → scale
// -----------------------------------------------------------------------
void WireframeApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    if (!hand.hand_detected) {
        rotX_ += spinVelX_;
        rotY_ += spinVelY_;
        rotZ_ += spinVelZ_;
        spinVelX_ *= SPIN_FRICTION;
        spinVelY_ *= SPIN_FRICTION;
        spinVelZ_ *= SPIN_FRICTION;
        if (fabsf(spinVelX_) + fabsf(spinVelY_) + fabsf(spinVelZ_) < 0.001f)
            spinVelY_ = DEFAULT_SPIN_Y;
        return;
    }

    // Index tip offset from centre drives rotation velocity
    float dx = hand.lm_x[8] - 0.5f;
    float dy = hand.lm_y[8] - 0.5f;
    spinVelX_ = -dy * SCALE_SPIN_MAX * 2.0f;
    spinVelY_ =  dx * SCALE_SPIN_MAX * 2.0f;
    spinVelZ_ = 0.0f;
    rotX_ += spinVelX_;
    rotY_ += spinVelY_;

    // Pinch distance drives scale
    float t = (landmarkDist(hand, 4, 8) - 0.03f) / 0.22f;
    t = std::max(0.0f, std::min(1.0f, t));
    scale_ += (0.3f + t * 1.7f - scale_) * 0.3f;
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
        voxpaint::rotateXYZ(rotX_, rotY_, rotZ_, v, rot);

        // Scale and map to voxel space
        tx[i] = (rot[0] * scale_ + 1.0f) * 0.5f * (VOXEL_W - 1);
        ty[i] = (rot[1] * scale_ + 1.0f) * 0.5f * (VOXEL_H - 1);
        tz[i] = (rot[2] * scale_ + 1.0f) * 0.5f * (VOXEL_D - 1);
    }

    // Paint all edges
    for (auto& edge : mesh_.edges) {
        int a = edge[0], b = edge[1];
        if (a < 0 || a >= numVerts || b < 0 || b >= numVerts) continue;

        voxpaint::paint3DLine(voxels,
                    (int)roundf(tx[a]), (int)roundf(ty[a]), (int)roundf(tz[a]),
                    (int)roundf(tx[b]), (int)roundf(ty[b]), (int)roundf(tz[b]),
                    R, G, B);
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
