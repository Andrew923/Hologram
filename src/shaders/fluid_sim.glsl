#version 450 core
// Height-field wave simulation + voxel fill.
// Each invocation handles one (x,z) column of the 128×128 height map.
// Reads hCur and hPrev, writes hNext and fills the 3-D voxel grid.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba8) uniform image3D uVoxelGrid;
layout(binding = 1, r32f)  uniform image2D uHeightCur;
layout(binding = 2, r32f)  uniform image2D uHeightPrev;
layout(binding = 3, r32f)  uniform image2D uHeightNext;

uniform vec2  uFingerXZ;
uniform float uImpulse;
uniform int   uFingerActive;

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------
const float REST_HEIGHT   = 8.0;
const float WAVE_SPEED2   = 0.22;  // c²; stability requires c² ≤ 0.5
const float DAMPING       = 0.995; // per-frame energy loss (mix toward rest)
const float FINGER_RADIUS = 10.0;  // influence radius in voxels
const int   Y_FLOOR       = 8;     // voxel Y where pool floor sits
const int   MAX_H         = 48;    // max voxels above floor (3× original 16)

const float CX      = 63.5;
const float CZ      = 63.5;
const float CORE_R2 = 196.0;  // (14 px)²

const vec4 BODY_COLOR    = vec4(0.000, 0.471, 0.784, 1.0);  // deep blue
const vec4 SURFACE_COLOR = vec4(0.235, 0.784, 1.000, 1.0);  // bright cyan
const vec4 ZERO          = vec4(0.0);

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
bool isUsable(ivec2 p) {
    if (p.x < 16 || p.x > 112 || p.y < 16 || p.y > 112) return false;
    float dx = float(p.x) - CX;
    float dz = float(p.y) - CZ;
    return (dx * dx + dz * dz) >= CORE_R2;
}

// Returns REST_HEIGHT for boundary/unusable cells so waves reflect
float sampleH(ivec2 p) {
    return isUsable(p) ? imageLoad(uHeightCur, p).r : REST_HEIGHT;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
void main() {
    ivec2 xz = ivec2(gl_GlobalInvocationID.xy);
    if (xz.x >= 128 || xz.y >= 128) return;

    float hNew;

    if (!isUsable(xz)) {
        // Dead zone: no water, write zero to height map
        hNew = 0.0;
    } else {
        float hC = imageLoad(uHeightCur,  xz).r;
        float hP = imageLoad(uHeightPrev, xz).r;

        // Discrete 2-D wave equation
        float lap = sampleH(xz + ivec2(-1,  0))
                  + sampleH(xz + ivec2( 1,  0))
                  + sampleH(xz + ivec2( 0, -1))
                  + sampleH(xz + ivec2( 0,  1))
                  - 4.0 * hC;

        hNew = 2.0 * hC - hP + WAVE_SPEED2 * lap;

        // Damp toward rest height
        hNew = mix(REST_HEIGHT, hNew, DAMPING);

        // Finger depression (repulsion pushes water away from fingertip)
        if (uFingerActive != 0) {
            float fdx = float(xz.x) - uFingerXZ.x;
            float fdz = float(xz.y) - uFingerXZ.y;
            float fr2 = fdx * fdx + fdz * fdz;
            float fR2 = FINGER_RADIUS * FINGER_RADIUS;
            if (fr2 < fR2)
                hNew -= uImpulse * (1.0 - fr2 / fR2);
        }

        hNew = clamp(hNew, 0.0, float(MAX_H));
    }

    imageStore(uHeightNext, xz, vec4(hNew, 0.0, 0.0, 0.0));

    // Fill voxel column: body below waterTop, surface at waterTop, zero above
    int waterTop = Y_FLOOR + int(round(hNew));
    bool usable  = isUsable(xz);

    for (int y = Y_FLOOR; y <= Y_FLOOR + MAX_H; y++) {
        vec4 col;
        if      (usable && y <  waterTop) col = BODY_COLOR;
        else if (usable && y == waterTop) col = SURFACE_COLOR;
        else                               col = ZERO;
        imageStore(uVoxelGrid, ivec3(xz.x, y, xz.y), col);
    }
}
