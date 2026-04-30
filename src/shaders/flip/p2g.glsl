#version 450 core
// Particle → Grid scatter. Each particle distributes its (vel * kernel_w) and
// kernel_w into 8 surrounding cell-centred grid samples via trilinear weights,
// using fixed-point integer atomics (no float-atomic extension required).
layout(local_size_x = 256) in;

struct Particle {
    vec4 posLife;
    vec4 velPad;
};

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

layout(binding = 1, r32i) uniform iimage3D aVelX;
layout(binding = 2, r32i) uniform iimage3D aVelY;
layout(binding = 3, r32i) uniform iimage3D aVelZ;
layout(binding = 4, r32i) uniform iimage3D aWeight;

uniform uint uParticleCount;

const float FIXED_SCALE = 65536.0;
const float GRID_TO_VOX = 2.0;

void splat(ivec3 c, float w, vec3 v) {
    if (any(lessThan(c, ivec3(0))) ||
        c.x >= 64 || c.y >= 32 || c.z >= 64) return;
    int wi = int(w * FIXED_SCALE);
    if (wi == 0) return;
    imageAtomicAdd(aWeight, c, wi);
    imageAtomicAdd(aVelX,   c, int(v.x * w * FIXED_SCALE));
    imageAtomicAdd(aVelY,   c, int(v.y * w * FIXED_SCALE));
    imageAtomicAdd(aVelZ,   c, int(v.z * w * FIXED_SCALE));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uParticleCount) return;

    Particle p = particles[idx];
    if (p.posLife.w < 0.0) return; // dead

    // Voxel space → grid space (cell-centred, 1 cell = 2 voxels)
    vec3 g = p.posLife.xyz / GRID_TO_VOX - 0.5;
    ivec3 g0 = ivec3(floor(g));
    vec3 f  = g - vec3(g0);

    vec3 v = p.velPad.xyz;

    // 8 trilinear weights
    float w000 = (1.0-f.x)*(1.0-f.y)*(1.0-f.z);
    float w100 = (    f.x)*(1.0-f.y)*(1.0-f.z);
    float w010 = (1.0-f.x)*(    f.y)*(1.0-f.z);
    float w110 = (    f.x)*(    f.y)*(1.0-f.z);
    float w001 = (1.0-f.x)*(1.0-f.y)*(    f.z);
    float w101 = (    f.x)*(1.0-f.y)*(    f.z);
    float w011 = (1.0-f.x)*(    f.y)*(    f.z);
    float w111 = (    f.x)*(    f.y)*(    f.z);

    splat(g0 + ivec3(0,0,0), w000, v);
    splat(g0 + ivec3(1,0,0), w100, v);
    splat(g0 + ivec3(0,1,0), w010, v);
    splat(g0 + ivec3(1,1,0), w110, v);
    splat(g0 + ivec3(0,0,1), w001, v);
    splat(g0 + ivec3(1,0,1), w101, v);
    splat(g0 + ivec3(0,1,1), w011, v);
    splat(g0 + ivec3(1,1,1), w111, v);
}
