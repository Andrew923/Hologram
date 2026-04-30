#version 450 core
// Seed/clear pass for the cellular automaton.
//   uClear=1 zeroes the grid first.
//   If uRadius > 0, paints a hashed ~50%-density sphere at uCenter on top.
// Used both for initial setup (clear + seed at centre) and for pinch-driven
// drips at the user's fingertip (no clear, small radius).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r8ui) uniform uimage3D uCells;

uniform ivec3 uCenter;
uniform float uRadius;
uniform uint  uSeed;
uniform uint  uClear;

uint hash(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    uint result = (uClear != 0u) ? 0u : imageLoad(uCells, c).r;

    if (uRadius > 0.0) {
        vec3 d = vec3(c - uCenter);
        if (dot(d, d) <= uRadius * uRadius) {
            uint h = hash(uint(c.x) * 73u + uint(c.y) * 17u + uint(c.z) * 31u + uSeed);
            if ((h & 0x7u) < 4u) result = 1u;
        }
    }

    imageStore(uCells, c, uvec4(result));
}
