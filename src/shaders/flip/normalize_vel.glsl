#version 450 core
// Decode fixed-point P2G accumulators into float velocities, normalize by
// weight, apply gravity tilt, and snapshot the post-gravity grid state for
// the FLIP delta computed later in G2P.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32i) uniform iimage3D aVelX;
layout(binding = 1, r32i) uniform iimage3D aVelY;
layout(binding = 2, r32i) uniform iimage3D aVelZ;
layout(binding = 3, r32i) uniform iimage3D aWeight;

layout(binding = 4, r32f) uniform image3D gVelX;
layout(binding = 5, r32f) uniform image3D gVelY;
layout(binding = 6, r32f) uniform image3D gVelZ;

layout(binding = 7, r32f) uniform image3D gVelXSave;
layout(binding = 8, r32f) uniform image3D gVelYSave;
layout(binding = 9, r32f) uniform image3D gVelZSave;

layout(binding = 10, r32f) uniform image3D gWeightF;

uniform vec3  uGravity;   // voxel/sec^2
uniform float uDt;

const float INV_FIXED = 1.0 / 65536.0;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    float w = float(imageLoad(aWeight, c).r) * INV_FIXED;
    vec3 v = vec3(0.0);
    if (w > 1e-4) {
        v.x = float(imageLoad(aVelX, c).r) * INV_FIXED / w;
        v.y = float(imageLoad(aVelY, c).r) * INV_FIXED / w;
        v.z = float(imageLoad(aVelZ, c).r) * INV_FIXED / w;
    }

    imageStore(gWeightF, c, vec4(w));

    // Apply gravity to fluid cells (where mass exists).
    if (w > 1e-4) v += uGravity * uDt;

    imageStore(gVelX, c, vec4(v.x));
    imageStore(gVelY, c, vec4(v.y));
    imageStore(gVelZ, c, vec4(v.z));

    // Snapshot AFTER gravity so the FLIP delta carries gravity-induced change.
    imageStore(gVelXSave, c, vec4(v.x));
    imageStore(gVelYSave, c, vec4(v.y));
    imageStore(gVelZSave, c, vec4(v.z));
}
