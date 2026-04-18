#pragma once
// -----------------------------------------------------------------------
// VoxelPaint — shared voxel painting primitives used across apps.
//
// Header-only. All apps that paint into the CPU-side voxel buffer
// (layout: data[(z*VOXEL_H + y)*VOXEL_W + x] * 4 bytes RGBA8) should use
// these instead of duplicating the implementation.
// -----------------------------------------------------------------------
#include "../engine/Renderer.h"    // VOXEL_W/H/D, VOXEL_BYTES
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace voxpaint {

// Paint a single voxel (RGBA8) at (x,y,z). Out-of-bounds writes are
// silently ignored.
inline void paintVoxel(uint8_t* voxels, int x, int y, int z,
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

// Paint a 3D line segment from (x0,y0,z0) to (x1,y1,z1) using DDA along
// the dominant axis. Matches the CubeApp implementation exactly.
inline void paint3DLine(uint8_t* voxels,
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
        int x = (int)std::round(x0 + t * (x1 - x0));
        int y = (int)std::round(y0 + t * (y1 - y0));
        int z = (int)std::round(z0 + t * (z1 - z0));
        paintVoxel(voxels, x, y, z, r, g, b);
    }
}

// Paint a solid axis-aligned voxel cube of half-extent `r` centered at
// (cx,cy,cz). Used for highlighting particle positions and the finger
// cursor.
inline void paintCube(uint8_t* voxels, int cx, int cy, int cz, int half,
                      uint8_t r, uint8_t g, uint8_t b)
{
    for (int dz = -half; dz <= half; ++dz)
        for (int dy = -half; dy <= half; ++dy)
            for (int dx = -half; dx <= half; ++dx)
                paintVoxel(voxels, cx + dx, cy + dy, cz + dz, r, g, b);
}

// Paint a filled sphere of given radius centered at (cx,cy,cz).
inline void paintSphere(uint8_t* voxels, int cx, int cy, int cz, float radius,
                        uint8_t r, uint8_t g, uint8_t b)
{
    int ir = (int)std::ceil(radius);
    float r2 = radius * radius;
    for (int dz = -ir; dz <= ir; ++dz)
        for (int dy = -ir; dy <= ir; ++dy)
            for (int dx = -ir; dx <= ir; ++dx)
                if ((float)(dx*dx + dy*dy + dz*dz) <= r2)
                    paintVoxel(voxels, cx + dx, cy + dy, cz + dz, r, g, b);
}

// Apply the combined rotation R = Rz * Ry * Rx (matching cube.py
// conventions) to vector v, storing the result in out. This is the same
// formula used by CubeApp::rotate and WireframeApp::rotate.
inline void rotateXYZ(float rx, float ry, float rz,
                      const float v[3], float out[3])
{
    float cx = std::cos(rx), sx = std::sin(rx);
    float cy = std::cos(ry), sy = std::sin(ry);
    float cz = std::cos(rz), sz = std::sin(rz);

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

}  // namespace voxpaint
