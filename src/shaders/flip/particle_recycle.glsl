#version 450 core
// Recycle dead (life < 0) particles. The buffer is split into two pools:
//
//   [0 .. uBaseCount)              — bowl pool: respawned at random in the
//                                    annular fluid region near the floor.
//   [uBaseCount .. uParticleCount) — pinch reserve: stays dead unless pinch
//                                    is active; then a small per-frame slice
//                                    is spawned at uPinchPos with downward
//                                    velocity. Once alive these particles
//                                    behave as normal fluid; if they exit the
//                                    bowl (life flips to <0 in g2p_advect) the
//                                    next pinch can re-use them.
layout(local_size_x = 256) in;

struct Particle {
    vec4 posLife;   // xyz: voxel-space position; w: life (>=0 alive, <0 dead)
    vec4 velPad;    // xyz: velocity (voxel/sec)
};

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

uniform uint  uParticleCount;
uniform uint  uBaseCount;
uniform uint  uFrame;
uniform float uFillHeight;

uniform uint  uPinchActive;       // 0 or 1
uniform vec3  uPinchPos;          // voxel-space spawn centre
uniform uint  uPinchSpawnStart;   // first pinch-pool offset to spawn this frame
uniform uint  uPinchSpawnCount;   // how many pinch-pool slots to spawn

const float CX        = 63.5;
const float CZ        = 63.5;
const float R_INNER   = 16.0;
const float R_OUTER   = 58.0;
const float Y_FLOOR   = 9.0;

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

    if (idx < uBaseCount) {
        // Bowl pool: existing behaviour.
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
        return;
    }

    // Pinch reserve.
    if (uPinchActive == 0u) return;

    uint poolSize = uParticleCount - uBaseCount;
    uint pinchIdx = idx - uBaseCount;
    // Spawn slice [uPinchSpawnStart, uPinchSpawnStart + uPinchSpawnCount), wrapped.
    uint dist = (pinchIdx + poolSize - uPinchSpawnStart) % poolSize;
    if (dist >= uPinchSpawnCount) return;

    float ox = (frand(seed) - 0.5) * 1.6;   // small radial scatter
    float oz = (frand(seed) - 0.5) * 1.6;
    float oy = (frand(seed) - 0.5) * 0.6;
    vec3 pos = uPinchPos + vec3(ox, oy, oz);

    // Initial velocity: light lateral spread + downward.
    float vx = (frand(seed) - 0.5) * 3.0;
    float vz = (frand(seed) - 0.5) * 3.0;
    float vy = -3.0 - frand(seed) * 5.0;    // -3 .. -8 voxel/sec

    particles[idx].posLife = vec4(pos, 1.0);
    particles[idx].velPad  = vec4(vx, vy, vz, 0.0);
}
