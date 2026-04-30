#version 450 core
// Splat each live particle as a single voxel into the 128×64×128 RGBA8
// display volume. Fast particles also leave a one-voxel motion-blur tail.
// Voxel volume is cleared by Renderer::clearVoxels() each frame, so a
// non-atomic imageStore is fine — visualization-level races are invisible
// at the slice scan rate.
layout(local_size_x = 256) in;

struct Particle {
    vec4 posLife;
    vec4 velPad;
};

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

layout(binding = 1, rgba8) uniform writeonly image3D uVoxelGrid;

uniform uint  uParticleCount;
uniform float uDt;

const vec4 COLOR_DEEP   = vec4(0.00, 0.30, 0.70, 1.0);
const vec4 COLOR_SPRAY  = vec4(0.90, 0.95, 1.00, 1.0);
const float V_SPRAY     = 12.0;   // velocity at which colour saturates
const float V_TAIL      = 6.0;    // motion-blur tail threshold

void plot(ivec3 p, vec4 col) {
    if (p.x < 0 || p.x >= 128 || p.y < 0 || p.y >= 64 || p.z < 0 || p.z >= 128) return;
    imageStore(uVoxelGrid, p, col);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uParticleCount) return;

    Particle p = particles[idx];
    if (p.posLife.w < 0.0) return;

    vec3 pos = p.posLife.xyz;
    vec3 v   = p.velPad.xyz;
    float speed = length(v);

    float t = clamp(speed / V_SPRAY, 0.0, 1.0);
    t = smoothstep(0.0, 1.0, t);
    vec4 col = mix(COLOR_DEEP, COLOR_SPRAY, t);

    ivec3 head = ivec3(floor(pos));
    plot(head, col);
    // Stack one voxel above so a thin settled layer still reads as volumetric
    // on the rotor (single-voxel splats render as a 1px ring otherwise).
    plot(head + ivec3(0, 1, 0), col);

    if (speed > V_TAIL) {
        vec3 tail = pos - 0.5 * v * uDt;
        ivec3 tailV = ivec3(floor(tail));
        if (tailV != head) plot(tailV, col);
    }
}
