#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include "../engine/ObjLoader.h"
#include <cstdint>
#include <string>

class WireframeApp : public IApplication {
public:
    WireframeApp() = default;
    ~WireframeApp() override = default;

    // Load/replace the OBJ model. Safe to call before or after setup().
    // Returns true on success. If called after setup(), geometry is
    // hot-swapped for the next frame.
    bool setModel(const std::string& objPath);

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    void teardown(Renderer&)                override {}
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    ReturnToMenuWatcher menuWatcher_;
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

    // Finger extension detection: true if fingertip is farther from wrist than MCP
    static bool isFingerExtended(const SharedHandData& hand, int tipIdx, int mcpIdx);
    // Distance between two landmarks
    static float landmarkDist(const SharedHandData& hand, int a, int b);
};
