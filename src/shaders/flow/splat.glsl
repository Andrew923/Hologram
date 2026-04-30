#version 450 core
// Splat each particle as a single voxel. World coords are mapped to voxel
// coords via uOrigin (subtracted) and uScale (multiplied), then centred on
// the (64, 32, 64) bowl midpoint.
layout(local_size_x = 256) in;

struct Particle { vec4 pos; };

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };
layout(binding = 1, rgba8) uniform writeonly image3D uVoxelGrid;

uniform uint  uCount;
uniform vec3  uOrigin;
uniform vec3  uScale;
uniform vec4  uColor;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uCount) return;

    vec3 p  = particles[idx].pos.xyz;
    vec3 vf = (p - uOrigin) * uScale + vec3(64.0, 32.0, 64.0);
    ivec3 v = ivec3(floor(vf));
    if (v.x < 0 || v.x >= 128 || v.y < 0 || v.y >= 64 || v.z < 0 || v.z >= 128) return;
    imageStore(uVoxelGrid, v, uColor);
}
