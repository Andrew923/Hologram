#version 450 core
// 3D cellular automaton step. Each cell counts its 26 Moore-neighbourhood
// neighbours and applies the B/S rule encoded as bitmasks (bit n set if
// n-neighbour count triggers birth or survival).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r8ui) uniform readonly  uimage3D uCur;
layout(binding = 1, r8ui) uniform writeonly uimage3D uNext;

uniform uint uBirth;
uniform uint uSurvive;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    uint count = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        ivec3 n = c + ivec3(dx, dy, dz);
        if (n.x < 0 || n.x >= 64 ||
            n.y < 0 || n.y >= 32 ||
            n.z < 0 || n.z >= 64) continue;
        if (imageLoad(uCur, n).r != 0u) count++;
    }

    bool alive = imageLoad(uCur, c).r != 0u;
    bool next  = alive ? ((uSurvive & (1u << count)) != 0u)
                       : ((uBirth   & (1u << count)) != 0u);
    imageStore(uNext, c, uvec4(next ? 1u : 0u));
}
