#version 450 core
// Snapshot the pre-gravity grid velocity into the save buffers so G2P can
// compute the FLIP delta as (v_new - v_pre_gravity) — capturing both gravity
// and the pressure correction. Gravity is then applied in-place to the live
// velocity field, which keeps the bug fix contained to this pass and stays
// within the 8-image-unit hardware limit (7 bindings used).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform image3D gVelX;
layout(binding = 1, r32f) uniform image3D gVelY;
layout(binding = 2, r32f) uniform image3D gVelZ;

layout(binding = 3, r32f) uniform writeonly image3D gVelXSave;
layout(binding = 4, r32f) uniform writeonly image3D gVelYSave;
layout(binding = 5, r32f) uniform writeonly image3D gVelZSave;

layout(binding = 6, r32f) uniform readonly  image3D gWeightF;

uniform vec3  uGravity;   // voxel/sec^2
uniform float uDt;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    float vx = imageLoad(gVelX, c).r;
    float vy = imageLoad(gVelY, c).r;
    float vz = imageLoad(gVelZ, c).r;

    imageStore(gVelXSave, c, vec4(vx));
    imageStore(gVelYSave, c, vec4(vy));
    imageStore(gVelZSave, c, vec4(vz));

    if (imageLoad(gWeightF, c).r > 1e-4) {
        imageStore(gVelX, c, vec4(vx + uGravity.x * uDt));
        imageStore(gVelY, c, vec4(vy + uGravity.y * uDt));
        imageStore(gVelZ, c, vec4(vz + uGravity.z * uDt));
    }
}
