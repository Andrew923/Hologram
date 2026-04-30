#version 450 core
// Decode fixed-point P2G accumulators into float velocities, normalize by
// weight, and write the pre-gravity grid velocities. Gravity is applied in
// snapshot_vel.glsl after the FLIP-delta snapshot is taken, so the snapshot
// captures the pre-gravity field and (vNew - vSave) in G2P includes both
// gravity and pressure correction.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32i) uniform iimage3D aVelX;
layout(binding = 1, r32i) uniform iimage3D aVelY;
layout(binding = 2, r32i) uniform iimage3D aVelZ;
layout(binding = 3, r32i) uniform iimage3D aWeight;

layout(binding = 4, r32f) uniform image3D gVelX;
layout(binding = 5, r32f) uniform image3D gVelY;
layout(binding = 6, r32f) uniform image3D gVelZ;

layout(binding = 7, r32f) uniform image3D gWeightF;

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
    imageStore(gVelX, c, vec4(v.x));
    imageStore(gVelY, c, vec4(v.y));
    imageStore(gVelZ, c, vec4(v.z));
}
