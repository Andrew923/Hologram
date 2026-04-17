#pragma once
#include "IApplication.h"
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
    float scale_  = 1.0f;   // clamped to ~16–32 px tall
    float posX_   = 0.0f;   // voxel-space X offset, clamped to ±8
    float posY_   = 0.0f;   // voxel-space Y offset, clamped to ±8
    float posZ_   = 0.0f;   // smoothed toward core-safe target in update()

};
