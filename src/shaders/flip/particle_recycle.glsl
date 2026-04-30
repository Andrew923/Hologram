#version 450 core
// Recycle particles flagged dead (life < 0) by respawning them at a random
// position inside the annular fluid region near the floor. Cheap PRNG keyed on
// particle index + frame number. Live particles untouched.
layout(local_size_x = 256) in;

struct Particle {
    vec4 posLife;   // xyz: voxel-space position; w: life (>=0 alive, <0 dead)
    vec4 velPad;    // xyz: velocity (voxel/sec)
};

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

uniform uint  uParticleCount;
uniform uint  uFrame;
uniform float uFillHeight;   // voxel units (e.g. 22.0)

const float CX        = 63.5;
const float CZ        = 63.5;
const float R_INNER   = 16.0;  // a little outside the dead-zone for safety
const float R_OUTER   = 58.0;  // a little inside the outer wall
const float Y_FLOOR   = 9.0;

// xorshift32-based hash
uint hash(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}
float frand(inout uint s) {
    s = hash(s);
    return float(s) * (1.0 / 4294967295.0);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uParticleCount) return;

    Particle p = particles[idx];
    if (p.posLife.w >= 0.0) return; // alive — no respawn

    uint seed = hash(idx * 747796405u + uFrame * 2891336453u + 1u);

    // Polar sample inside the annulus, biased toward outer ring for variety.
    float u1 = frand(seed);
    float u2 = frand(seed);
    float u3 = frand(seed);

    float r2     = mix(R_INNER * R_INNER, R_OUTER * R_OUTER, u1);
    float radius = sqrt(r2);
    float theta  = u2 * 6.2831853;
    float yJit   = u3 * (uFillHeight - 0.5) + Y_FLOOR + 0.25;

    vec3 pos = vec3(CX + radius * cos(theta),
                    yJit,
                    CZ + radius * sin(theta));

    particles[idx].posLife = vec4(pos, 1.0);
    particles[idx].velPad  = vec4(0.0);
}
