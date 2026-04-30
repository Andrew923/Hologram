#version 450 core
// One damped-Jacobi iteration of the pressure Poisson equation on a
// cell-centred grid. SOLID neighbours: Neumann (mirror own pressure).
// AIR neighbours: Dirichlet (p = 0). Damping ω = 0.67 stabilises convergence
// at the cost of a constant slowdown — fine for our 40 iterations.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32f) uniform image3D  gPIn;
layout(binding = 1, r32f) uniform image3D  gPOut;
layout(binding = 2, r32f) uniform image3D  gDivergence;
layout(binding = 3, r8ui) uniform uimage3D gCellType;

const float OMEGA = 0.67;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    uint t = imageLoad(gCellType, c).r;
    if (t != 0u) {
        // Non-fluid cells hold p = 0 (AIR sets the Dirichlet boundary;
        // SOLID is unused — neighbours mirror via the loop below).
        imageStore(gPOut, c, vec4(0.0));
        return;
    }

    float pSelf = imageLoad(gPIn, c).r;
    float pSum  = 0.0;

    ivec3 ns[6] = ivec3[6](
        ivec3(c.x + 1, c.y, c.z), ivec3(c.x - 1, c.y, c.z),
        ivec3(c.x, c.y + 1, c.z), ivec3(c.x, c.y - 1, c.z),
        ivec3(c.x, c.y, c.z + 1), ivec3(c.x, c.y, c.z - 1));

    for (int i = 0; i < 6; ++i) {
        ivec3 n = ns[i];
        bool oob = any(lessThan(n, ivec3(0))) ||
                   n.x >= 64 || n.y >= 32 || n.z >= 64;
        if (oob) {
            pSum += pSelf; // treat outside as solid Neumann
            continue;
        }
        uint nt = imageLoad(gCellType, n).r;
        if (nt == 2u)      pSum += pSelf;             // SOLID: Neumann
        else if (nt == 1u) pSum += 0.0;               // AIR: Dirichlet
        else               pSum += imageLoad(gPIn, n).r; // FLUID
    }

    float div = imageLoad(gDivergence, c).r;
    float pNew = (pSum - div) / 6.0;
    pNew = mix(pSelf, pNew, OMEGA);
    imageStore(gPOut, c, vec4(pNew));
}
