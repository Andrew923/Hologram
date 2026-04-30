#version 450 core
// One-step RK2 integration of a chaotic 3D ODE for each particle. Several
// substeps per dispatch keep the trajectory smooth even at low frame rates.
layout(local_size_x = 256) in;

struct Particle { vec4 pos; };  // xyz: world coords, w: age

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

uniform uint  uCount;
uniform float uDt;
uniform int   uSubsteps;
uniform int   uAttractor;

// Lorenz: σ=10, ρ=28, β=8/3
vec3 d_lorenz(vec3 p) {
    return vec3(10.0 * (p.y - p.x),
                p.x * (28.0 - p.z) - p.y,
                p.x * p.y - p.z * (8.0 / 3.0));
}

// Aizawa: a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1
vec3 d_aizawa(vec3 p) {
    return vec3((p.z - 0.7) * p.x - 3.5 * p.y,
                3.5 * p.x + (p.z - 0.7) * p.y,
                0.6 + 0.95 * p.z - p.z*p.z*p.z/3.0
                    - (p.x*p.x + p.y*p.y) * (1.0 + 0.25 * p.z)
                    + 0.1 * p.z * p.x*p.x*p.x);
}

// Halvorsen: a=1.4
vec3 d_halvorsen(vec3 p) {
    return vec3(-1.4 * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y,
                -1.4 * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z,
                -1.4 * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x);
}

// Thomas: b=0.208186
vec3 d_thomas(vec3 p) {
    return vec3(sin(p.y) - 0.208186 * p.x,
                sin(p.z) - 0.208186 * p.y,
                sin(p.x) - 0.208186 * p.z);
}

vec3 deriv(vec3 p, int a) {
    if      (a == 0) return d_lorenz(p);
    else if (a == 1) return d_aizawa(p);
    else if (a == 2) return d_halvorsen(p);
    else             return d_thomas(p);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uCount) return;

    vec3 p = particles[idx].pos.xyz;
    for (int i = 0; i < uSubsteps; ++i) {
        vec3 k1 = deriv(p, uAttractor);
        vec3 k2 = deriv(p + 0.5 * uDt * k1, uAttractor);
        p += uDt * k2;
    }
    particles[idx].pos.xyz = p;
    particles[idx].pos.w  += float(uSubsteps) * uDt;
}
