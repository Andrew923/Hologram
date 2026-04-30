#version 450 core
// Grid → Particle gather. Reads the post-pressure grid velocity (vNew) and
// the pre-pressure snapshot (vSave) at the particle's location. Updates the
// particle velocity with a FLIP/PIC blend, then advects with RK2 and resolves
// collisions against the annular-cylinder SDF.
layout(local_size_x = 256) in;

struct Particle {
    vec4 posLife;
    vec4 velPad;
};

layout(std430, binding = 0) buffer ParticleSSBO { Particle particles[]; };

layout(binding = 0, r32f) uniform readonly image3D gWeightF;
layout(binding = 1, r32f) uniform image3D gVelX;
layout(binding = 2, r32f) uniform image3D gVelY;
layout(binding = 3, r32f) uniform image3D gVelZ;
layout(binding = 4, r32f) uniform image3D gVelXSave;
layout(binding = 5, r32f) uniform image3D gVelYSave;
layout(binding = 6, r32f) uniform image3D gVelZSave;

uniform uint  uParticleCount;
uniform float uDt;
uniform float uFlipAlpha;

// Anti-compression buoyancy: FLIP/PIC volume drifts toward zero over time
// because trilinear gather can't perfectly preserve incompressibility, so the
// fluid pancakes onto the floor. Sampling local weight (≈ particles per cell)
// and adding an upward kick wherever the cell is over-dense gives the solver
// a steady volume target to fight gravity against. Excess is capped so that
// transient high-density events (pinch-spawn streams) don't launch particles
// through the ceiling.
const float W_TARGET     = 4.0;   // ~4 particles per grid cell at rest
const float W_KICK       = 6.0;   // voxel/sec^2 per excess particle
const float W_MAX_EXCESS = 4.0;   // cap so dense pinch streams don't blast up

const float GRID_TO_VOX = 2.0;

// World geometry in voxel units (matches mark_cells / slicer).
const float CX        = 63.5;
const float CZ        = 63.5;
const float R_INNER   = 14.5;
const float R_OUTER   = 61.5;
const float Y_FLOOR   = 8.0;
const float Y_CEIL    = 63.5;
const float RESTITUTION = 0.4;

// Trilinear sample of a single r32f image at grid-space coords. The image
// uniform is bound by name at each call site so the format qualifier matches.
#define TRILERP(IMG, G, OUT) \
    do { \
        vec3 _g = clamp((G), vec3(0.0), vec3(63.0, 31.0, 63.0)); \
        ivec3 _g0 = ivec3(floor(_g)); \
        vec3 _f = _g - vec3(_g0); \
        ivec3 _g1 = min(_g0 + ivec3(1), ivec3(63, 31, 63)); \
        float _v000 = imageLoad(IMG, ivec3(_g0.x, _g0.y, _g0.z)).r; \
        float _v100 = imageLoad(IMG, ivec3(_g1.x, _g0.y, _g0.z)).r; \
        float _v010 = imageLoad(IMG, ivec3(_g0.x, _g1.y, _g0.z)).r; \
        float _v110 = imageLoad(IMG, ivec3(_g1.x, _g1.y, _g0.z)).r; \
        float _v001 = imageLoad(IMG, ivec3(_g0.x, _g0.y, _g1.z)).r; \
        float _v101 = imageLoad(IMG, ivec3(_g1.x, _g0.y, _g1.z)).r; \
        float _v011 = imageLoad(IMG, ivec3(_g0.x, _g1.y, _g1.z)).r; \
        float _v111 = imageLoad(IMG, ivec3(_g1.x, _g1.y, _g1.z)).r; \
        float _v00 = mix(_v000, _v100, _f.x); \
        float _v10 = mix(_v010, _v110, _f.x); \
        float _v01 = mix(_v001, _v101, _f.x); \
        float _v11 = mix(_v011, _v111, _f.x); \
        float _v0  = mix(_v00, _v10, _f.y); \
        float _v1  = mix(_v01, _v11, _f.y); \
        (OUT) = mix(_v0, _v1, _f.z); \
    } while (false)

vec3 gridSampleNew(vec3 g) {
    float ax, ay, az;
    TRILERP(gVelX, g, ax);
    TRILERP(gVelY, g, ay);
    TRILERP(gVelZ, g, az);
    return vec3(ax, ay, az);
}
vec3 gridSampleSave(vec3 g) {
    float ax, ay, az;
    TRILERP(gVelXSave, g, ax);
    TRILERP(gVelYSave, g, ay);
    TRILERP(gVelZSave, g, az);
    return vec3(ax, ay, az);
}

// Annular cylinder SDF: positive outside (penetration depth), negative inside.
// Returns sdf value and outward normal.
void cylinderSDF(vec3 p, out float sdf, out vec3 n) {
    vec2 r = p.xz - vec2(CX, CZ);
    float dist = length(r);
    vec2 dirOut = (dist > 1e-4) ? (r / dist) : vec2(1.0, 0.0);

    float sOuter = dist - R_OUTER;        // > 0 outside outer wall
    float sInner = R_INNER - dist;        // > 0 inside dead zone
    float sFloor = Y_FLOOR - p.y;         // > 0 below floor

    sdf = max(max(sOuter, sInner), sFloor);
    if (sdf == sOuter)      n = vec3(dirOut.x, 0.0, dirOut.y);
    else if (sdf == sInner) n = vec3(-dirOut.x, 0.0, -dirOut.y);
    else                    n = vec3(0.0, -1.0, 0.0);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uParticleCount) return;

    Particle p = particles[idx];
    if (p.posLife.w < 0.0) return;

    vec3 pos = p.posLife.xyz;
    vec3 vp  = p.velPad.xyz;

    vec3 g = pos / GRID_TO_VOX - 0.5;
    vec3 vNew  = gridSampleNew(g);
    vec3 vSave = gridSampleSave(g);

    vec3 picVel  = vNew;
    vec3 flipVel = vp + (vNew - vSave);
    vp = mix(picVel, flipVel, uFlipAlpha);

    // Anti-compression buoyancy: point-sample the weight texture; over-dense
    // cells push particles upward to counteract FLIP volume drift.
    ivec3 wc = ivec3(clamp(g, vec3(0.0), vec3(63.0, 31.0, 63.0)));
    float w = imageLoad(gWeightF, wc).r;
    if (w > W_TARGET) vp.y += W_KICK * min(w - W_TARGET, W_MAX_EXCESS) * uDt;

    // RK2 (midpoint) advection in voxel units.
    vec3 mid = pos + 0.5 * uDt * vp;
    vec3 gMid = mid / GRID_TO_VOX - 0.5;
    vec3 vMid = gridSampleNew(gMid);
    pos += uDt * mix(vMid, vp, uFlipAlpha);

    // Collide against annulus walls / floor (analytic SDF).
    float sdf; vec3 n;
    cylinderSDF(pos, sdf, n);
    if (sdf > 0.0) {
        pos -= sdf * n;
        float vn = dot(vp, n);
        if (vn > 0.0) vp -= (1.0 + RESTITUTION) * vn * n;
    }

    // Open top: anything that escapes upward gets recycled next frame.
    float life = 1.0;
    if (pos.y > Y_CEIL) life = -1.0;

    particles[idx].posLife = vec4(pos, life);
    particles[idx].velPad  = vec4(vp, 0.0);
}
