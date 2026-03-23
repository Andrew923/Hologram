#pragma once
#include "IApplication.h"
#include "../engine/ObjLoader.h"
#include <cstdint>
#include <string>

class WireframeApp : public IApplication {
public:
    explicit WireframeApp(const std::string& objPath = "");
    ~WireframeApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    void teardown(Renderer&)                override {}
    bool bypassSlicer() const               override { return false; }

private:
    std::string objPath_;
    ObjMesh mesh_;

    // --- Mesh bounding box (computed once after load) ---
    float bboxMin_[3] = {0,0,0};
    float bboxMax_[3] = {0,0,0};
    float bboxCenter_[3] = {0,0,0};
    float bboxScale_ = 1.0f;   // normalizes model to [-1, 1]

    // --- Transformation state ---
    float rotX_   = 0.0f;
    float rotY_   = 0.0f;
    float rotZ_   = 0.0f;
    float scale_  = 1.0f;

    // Angular velocity for auto-spin / gesture-driven spin (radians per frame)
    float spinVelX_ = 0.0f;
    float spinVelY_ = 0.02f;   // default: slow Y auto-spin
    float spinVelZ_ = 0.0f;

    // --- Gesture detection state ---
    bool rotationActive_ = false;
    bool scaleActive_    = false;
    bool handPresent_    = false;

    // --- Helpers ---
    void computeBBox();
    void rotate(const float v[3], float out[3]) const;

    // Finger extension detection: true if fingertip is farther from wrist than MCP
    static bool isFingerExtended(const SharedHandData& hand, int tipIdx, int mcpIdx);
    // Distance between two landmarks
    static float landmarkDist(const SharedHandData& hand, int a, int b);

    // Voxel painting (reused from CubeApp pattern)
    static void paintVoxel(uint8_t* voxels, int x, int y, int z,
                           uint8_t r, uint8_t g, uint8_t b);
    static void paint3DLine(uint8_t* voxels,
                            int x0, int y0, int z0,
                            int x1, int y1, int z1,
                            uint8_t r, uint8_t g, uint8_t b);
};
