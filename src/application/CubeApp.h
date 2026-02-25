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
    // Current smoothed rotation angles (radians) and scale
    float rotX_   = 0.0f;
    float rotY_   = 0.0f;
    float rotZ_   = 0.0f;
    float scale_  = 1.0f;   // 0.3 – 2.0 range

    // Apply rotation matrix (Rz*Ry*Rx) to a vertex, return rotated x,y,z
    void rotate(const float v[3], float out[3]) const;

    // Paint a 3D line segment into the voxel buffer using DDA
    static void paint3DLine(uint8_t* voxels,
                            int x0, int y0, int z0,
                            int x1, int y1, int z1,
                            uint8_t r, uint8_t g, uint8_t b);

    static void paintVoxel(uint8_t* voxels, int x, int y, int z,
                           uint8_t r, uint8_t g, uint8_t b);
};
