#pragma once
#include "IApplication.h"
#include "../engine/CameraConfig.h"
#include <cstdint>

class CubeApp : public IApplication {
public:
    CubeApp() = default;
    ~CubeApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    void teardown(Renderer&)                override {}
    bool bypassSlicer() const               override { return false; }

private:
    // Current smoothed rotation angles (radians), scale, and position offset
    float rotX_   = 0.0f;
    float rotY_   = 0.0f;
    float rotZ_   = 0.0f;
    float scale_  = 1.0f;
    float posX_   = 0.0f;   // voxel-space X offset from center
    float posY_   = 0.0f;   // voxel-space Y offset from center
    float posZ_   = 0.0f;   // voxel-space Z offset from center

    // Camera config and depth estimation for Z
    CameraConfig cam_;
    bool         camOk_     = false;
    float        smoothedZ_ = 64.0f;

};
