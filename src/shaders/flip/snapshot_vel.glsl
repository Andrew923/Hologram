#version 450 core
// Copy the post-gravity grid velocity into the save buffers so G2P can
// compute the FLIP delta (v_new - v_old).  Split from normalize_vel to
// stay within the 8-image-unit hardware limit.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform readonly  image3D gVelX;
layout(binding = 1, r32f) uniform readonly  image3D gVelY;
layout(binding = 2, r32f) uniform readonly  image3D gVelZ;

layout(binding = 3, r32f) uniform writeonly image3D gVelXSave;
layout(binding = 4, r32f) uniform writeonly image3D gVelYSave;
layout(binding = 5, r32f) uniform writeonly image3D gVelZSave;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;
    imageStore(gVelXSave, c, imageLoad(gVelX, c));
    imageStore(gVelYSave, c, imageLoad(gVelY, c));
    imageStore(gVelZSave, c, imageLoad(gVelZ, c));
}
