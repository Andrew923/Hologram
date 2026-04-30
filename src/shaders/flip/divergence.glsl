#version 450 core
// Compute velocity divergence on a cell-centred grid using central differences.
// SOLID neighbours contribute 0 velocity (free-slip wall); AIR neighbours pass
// through. Result is the right-hand side for the pressure Poisson equation.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform image3D  gVelX;
layout(binding = 1, r32f) uniform image3D  gVelY;
layout(binding = 2, r32f) uniform image3D  gVelZ;
layout(binding = 3, r8ui) uniform uimage3D gCellType;
layout(binding = 4, r32f) uniform image3D  gDivergence;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    uint t = imageLoad(gCellType, c).r;
    if (t != 0u) {
        imageStore(gDivergence, c, vec4(0.0));
        return;
    }

    ivec3 cxp = ivec3(min(c.x + 1, 63), c.y, c.z);
    ivec3 cxm = ivec3(max(c.x - 1, 0),  c.y, c.z);
    ivec3 cyp = ivec3(c.x, min(c.y + 1, 31), c.z);
    ivec3 cym = ivec3(c.x, max(c.y - 1, 0),  c.z);
    ivec3 czp = ivec3(c.x, c.y, min(c.z + 1, 63));
    ivec3 czm = ivec3(c.x, c.y, max(c.z - 1, 0));

    uint txp = imageLoad(gCellType, cxp).r;
    uint txm = imageLoad(gCellType, cxm).r;
    uint typ = imageLoad(gCellType, cyp).r;
    uint tym = imageLoad(gCellType, cym).r;
    uint tzp = imageLoad(gCellType, czp).r;
    uint tzm = imageLoad(gCellType, czm).r;

    // SOLID neighbours contribute zero (free-slip wall).
    float vxp = (txp == 2u) ? 0.0 : imageLoad(gVelX, cxp).r;
    float vxm = (txm == 2u) ? 0.0 : imageLoad(gVelX, cxm).r;
    float vyp = (typ == 2u) ? 0.0 : imageLoad(gVelY, cyp).r;
    float vym = (tym == 2u) ? 0.0 : imageLoad(gVelY, cym).r;
    float vzp = (tzp == 2u) ? 0.0 : imageLoad(gVelZ, czp).r;
    float vzm = (tzm == 2u) ? 0.0 : imageLoad(gVelZ, czm).r;

    float div = 0.5 * ((vxp - vxm) + (vyp - vym) + (vzp - vzm));
    imageStore(gDivergence, c, vec4(div));
}
