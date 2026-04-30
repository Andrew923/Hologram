#include "ParticleApp.h"
#include "VoxelPaint.h"
#include "DisplayConstraints.h"
#include "GestureDetector.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr float DAMPING          = 0.98f;
static constexpr float WALL_STRENGTH    = 0.15f;
static constexpr float FINGER_STRENGTH  = 200.0f;   // force * 1/r^2 coefficient
static constexpr float FINGER_MIN_R2    = 9.0f;     // min distance squared
static constexpr float NEIGHBOR_RADIUS  = 4.0f;
static constexpr float NEIGHBOR_RADIUS2 = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
static constexpr float NEIGHBOR_STRENGTH = 0.35f;

// Working volume (inside the voxel grid).
static constexpr float X_MIN = 16.0f, X_MAX = 112.0f;
static constexpr float Y_MIN =  8.0f, Y_MAX =  56.0f;
static constexpr float Z_MIN = 16.0f, Z_MAX = 112.0f;

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float frandRange(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static inline void pushOutsideCoreRadius(float& x, float& z, float* vx = nullptr, float* vz = nullptr)
{
    float dx = x - 0.5f * (VOXEL_W - 1);
    float dz = z - 0.5f * (VOXEL_D - 1);
    float r2 = dx * dx + dz * dz;
    float minR2 = CORE_SAFE_RADIUS_PX * CORE_SAFE_RADIUS_PX;
    if (r2 >= minR2) return;

    float r = sqrtf(r2);
    float ux = 1.0f, uz = 0.0f;
    if (r > 1e-5f) {
        ux = dx / r;
        uz = dz / r;
    }

    x = 0.5f * (VOXEL_W - 1) + ux * CORE_SAFE_RADIUS_PX;
    z = 0.5f * (VOXEL_D - 1) + uz * CORE_SAFE_RADIUS_PX;

    if (vx && vz) {
        float vn = (*vx) * ux + (*vz) * uz;
        if (vn < 0.0f) {
            // remove inward radial velocity, keep tangential motion
            *vx -= vn * ux;
            *vz -= vn * uz;
        }
    }
}

void ParticleApp::resetParticles()
{
    // Warm/cool palette.
    static const uint8_t palette[6][3] = {
        {255,  80,  40},   // warm orange
        {255, 180,  30},   // amber
        {80,  220, 255},   // ice blue
        {160, 100, 255},   // violet
        {50,  255, 180},   // mint
        {255, 100, 180},   // pink
    };

    for (int i = 0; i < N_PARTICLES; ++i) {
        particles_[i].x  = frandRange(X_MIN + 8.0f, X_MAX - 8.0f);
        particles_[i].y  = frandRange(Y_MIN + 4.0f, Y_MAX - 4.0f);
        particles_[i].z  = frandRange(Z_MIN + 8.0f, Z_MAX - 8.0f);
        pushOutsideCoreRadius(particles_[i].x, particles_[i].z);
        particles_[i].vx   = frandRange(-0.3f, 0.3f);
        particles_[i].vy   = frandRange(-0.2f, 0.2f);
        particles_[i].vz   = frandRange(-0.3f, 0.3f);
        particles_[i].size = frandRange(1.0f, 2.5f);
        const uint8_t* c = palette[i % 6];
        particles_[i].r = c[0];
        particles_[i].g = c[1];
        particles_[i].b = c[2];
    }
}

void ParticleApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
    srand(12345);  // deterministic starting layout
    resetParticles();
}

void ParticleApp::computeCursor(const SharedHandData& hand)
{
    cursorValid_ = false;
    if (!hand.hand_detected) return;

    // Camera X → voxel X, camera Y → voxel Z (horizontal plane)
    // Pinch distance controls cursor Y (vertical height).
    curX_ = clampf(hand.lm_x[8] * (float)VOXEL_W, X_MIN, X_MAX);
    curZ_ = clampf(hand.lm_y[8] * (float)VOXEL_D, Z_MIN, Z_MAX);

    float pinch = hypotf(hand.lm_x[4] - hand.lm_x[8],
                         hand.lm_y[4] - hand.lm_y[8]);
    curY_ = Y_MIN + clampf((pinch - 0.03f) / 0.22f, 0.0f, 1.0f) * (Y_MAX - Y_MIN);
    pushOutsideCoreRadius(curX_, curZ_);
    cursorValid_ = true;
}

void ParticleApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);
    computeCursor(hand);

    Gesture g = detectGesture(hand);

    // Determine force sign: POINT attracts, FIST repels, else no force.
    float forceSign = 0.0f;
    if (cursorValid_) {
        if      (g == Gesture::POINT) forceSign = +1.0f;
        else if (g == Gesture::FIST)  forceSign = -1.0f;
    }

    // Spawn a particle on PINCH, rate-limited.
    if (spawnCooldown_ > 0) spawnCooldown_--;
    if (cursorValid_ && g == Gesture::PINCH && spawnCooldown_ == 0) {
        // Overwrite a random slot to keep the array bounded.
        int i = rand() % N_PARTICLES;
        particles_[i].x = curX_;
        particles_[i].y = curY_;
        particles_[i].z = curZ_;
        particles_[i].vx   = frandRange(-0.5f, 0.5f);
        particles_[i].vy   = frandRange(-0.5f, 0.5f);
        particles_[i].vz   = frandRange(-0.5f, 0.5f);
        particles_[i].size = frandRange(1.0f, 2.5f);
        pushOutsideCoreRadius(particles_[i].x, particles_[i].z);
        spawnCooldown_ = 10;
    }

    // --- Per-particle forces ---
    for (int i = 0; i < N_PARTICLES; ++i) {
        Particle& p = particles_[i];

        float ax = 0, ay = 0, az = 0;

        // Finger force (inverse square, softened).
        if (forceSign != 0.0f) {
            float dx = curX_ - p.x;
            float dy = curY_ - p.y;
            float dz = curZ_ - p.z;
            float r2 = dx*dx + dy*dy + dz*dz + FINGER_MIN_R2;
            float inv = FINGER_STRENGTH / (r2 * sqrtf(r2));
            ax += forceSign * dx * inv;
            ay += forceSign * dy * inv;
            az += forceSign * dz * inv;
        }

        // Soft wall: restoring force toward the nearest interior point.
        if      (p.x < X_MIN) ax += (X_MIN - p.x) * WALL_STRENGTH;
        else if (p.x > X_MAX) ax += (X_MAX - p.x) * WALL_STRENGTH;
        if      (p.y < Y_MIN) ay += (Y_MIN - p.y) * WALL_STRENGTH;
        else if (p.y > Y_MAX) ay += (Y_MAX - p.y) * WALL_STRENGTH;
        if      (p.z < Z_MIN) az += (Z_MIN - p.z) * WALL_STRENGTH;
        else if (p.z > Z_MAX) az += (Z_MAX - p.z) * WALL_STRENGTH;

        p.vx = (p.vx + ax) * DAMPING;
        p.vy = (p.vy + ay) * DAMPING;
        p.vz = (p.vz + az) * DAMPING;
    }

    // --- Inter-particle repulsion (O(N^2) but N=256, ~65k ops, fine) ---
    for (int i = 0; i < N_PARTICLES; ++i) {
        Particle& a = particles_[i];
        for (int j = i + 1; j < N_PARTICLES; ++j) {
            Particle& b = particles_[j];
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > NEIGHBOR_RADIUS2 || r2 < 1e-4f) continue;
            float r  = sqrtf(r2);
            float f  = NEIGHBOR_STRENGTH * (NEIGHBOR_RADIUS - r) / r;
            float fx = dx * f, fy = dy * f, fz = dz * f;
            a.vx += fx; a.vy += fy; a.vz += fz;
            b.vx -= fx; b.vy -= fy; b.vz -= fz;
        }
    }

    // --- Integrate ---
    for (int i = 0; i < N_PARTICLES; ++i) {
        Particle& p = particles_[i];
        p.x += p.vx;
        p.y += p.vy;
        p.z += p.vz;
        pushOutsideCoreRadius(p.x, p.z, &p.vx, &p.vz);
    }
}

void ParticleApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    // Paint particles as spheres with per-particle size variation.
    for (int i = 0; i < N_PARTICLES; ++i) {
        const Particle& p = particles_[i];
        int ix = (int)roundf(p.x);
        int iy = (int)roundf(p.y);
        int iz = (int)roundf(p.z);
        voxpaint::paintSphere(voxels, ix, iy, iz, p.size, p.r, p.g, p.b);
    }

    // Paint red cursor cube at the fingertip.
    if (cursorValid_) {
        voxpaint::paintCube(voxels,
                            (int)roundf(curX_),
                            (int)roundf(curY_),
                            (int)roundf(curZ_),
                            1,       // 3x3x3 cube
                            255, 0, 0);
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
