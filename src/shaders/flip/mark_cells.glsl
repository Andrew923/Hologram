#version 450 core
// Set per-cell type for the pressure solve:
//   0 = FLUID  (weight > eps and inside the annular bowl)
//   1 = AIR    (free surface — Dirichlet p = 0)
//   2 = SOLID  (outside annulus, below floor — Neumann)
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform image3D  gWeightF;
layout(binding = 1, r8ui) uniform uimage3D gCellType;

// Cylinder geometry in *voxel* units; convert to grid units (1 cell = 2 voxels).
const float CX_G    = 31.75; // (63.5 / 2)
const float CZ_G    = 31.75;
const float R_IN_G  = 7.0;   // inner dead-zone radius (~14 voxels)
const float R_OUT_G = 31.0;  // outer wall radius (~62 voxels) keeps margin
const float Y_FLOOR_G = 4.0; // 8 voxels / 2

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    vec3 cc = vec3(c) + 0.5;
    float dx = cc.x - CX_G;
    float dz = cc.z - CZ_G;
    float r2 = dx*dx + dz*dz;

    bool solid = (r2 < R_IN_G * R_IN_G)  ||
                 (r2 > R_OUT_G * R_OUT_G) ||
                 (cc.y < Y_FLOOR_G);

    uint type;
    if (solid) {
        type = 2u;
    } else {
        float w = imageLoad(gWeightF, c).r;
        type = (w > 0.05) ? 0u : 1u;
    }
    imageStore(gCellType, c, uvec4(type));
}
