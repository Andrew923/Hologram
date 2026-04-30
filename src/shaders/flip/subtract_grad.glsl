#version 450 core
// Project the velocity field divergence-free by subtracting ∇p (central
// differences). Only fluid cells are updated; solid cells stay at zero so
// that the G2P interpolation sees a clean wall.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform image3D  gVelX;
layout(binding = 1, r32f) uniform image3D  gVelY;
layout(binding = 2, r32f) uniform image3D  gVelZ;
layout(binding = 3, r32f) uniform image3D  gP;
layout(binding = 4, r8ui) uniform uimage3D gCellType;

float pAt(ivec3 p, float pSelf) {
    bool oob = any(lessThan(p, ivec3(0))) ||
               p.x >= 64 || p.y >= 32 || p.z >= 64;
    if (oob) return pSelf;
    uint t = imageLoad(gCellType, p).r;
    if (t == 2u) return pSelf; // SOLID Neumann
    if (t == 1u) return 0.0;   // AIR Dirichlet
    return imageLoad(gP, p).r;
}

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    uint t = imageLoad(gCellType, c).r;
    if (t != 0u) {
        imageStore(gVelX, c, vec4(0.0));
        imageStore(gVelY, c, vec4(0.0));
        imageStore(gVelZ, c, vec4(0.0));
        return;
    }

    float pSelf = imageLoad(gP, c).r;
    float pxp = pAt(ivec3(c.x + 1, c.y, c.z), pSelf);
    float pxm = pAt(ivec3(c.x - 1, c.y, c.z), pSelf);
    float pyp = pAt(ivec3(c.x, c.y + 1, c.z), pSelf);
    float pym = pAt(ivec3(c.x, c.y - 1, c.z), pSelf);
    float pzp = pAt(ivec3(c.x, c.y, c.z + 1), pSelf);
    float pzm = pAt(ivec3(c.x, c.y, c.z - 1), pSelf);

    float vx = imageLoad(gVelX, c).r - 0.5 * (pxp - pxm);
    float vy = imageLoad(gVelY, c).r - 0.5 * (pyp - pym);
    float vz = imageLoad(gVelZ, c).r - 0.5 * (pzp - pzm);

    imageStore(gVelX, c, vec4(vx));
    imageStore(gVelY, c, vec4(vy));
    imageStore(gVelZ, c, vec4(vz));
}
