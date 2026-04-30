#include "FluidApp.h"
#include "../engine/Renderer.h"
#include <epoxy/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Particle struct must match GLSL `Particle { vec4 posLife; vec4 velPad; }`.
struct ParticleCpu {
    float pos[3];
    float life;
    float vel[3];
    float pad;
};
static_assert(sizeof(ParticleCpu) == 32, "ParticleCpu layout mismatch");

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------
// Compile + link a single-stage compute shader.
// -----------------------------------------------------------------------
bool FluidApp::loadProgram(const char* path, GLuint& outProg)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FluidApp: cannot open shader: %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(len + 1);
    fread(src, 1, len, f);
    fclose(f);
    src[len] = '\0';

    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, (const GLchar**)&src, nullptr);
    glCompileShader(cs);
    free(src);

    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(cs, sizeof(log), nullptr, log);
        fprintf(stderr, "FluidApp: shader compile error (%s):\n%s\n", path, log);
        glDeleteShader(cs);
        return false;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glDeleteShader(cs);

    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "FluidApp: shader link error (%s):\n%s\n", path, log);
        glDeleteProgram(prog);
        return false;
    }
    outProg = prog;
    return true;
}

// -----------------------------------------------------------------------
// Allocate one 3D image of given internal format.
// -----------------------------------------------------------------------
static GLuint makeGridTex(GLenum internalFormat)
{
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_3D, t);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexStorage3D(GL_TEXTURE_3D, 1, internalFormat,
                   FluidApp::GRID_W, FluidApp::GRID_H, FluidApp::GRID_D);
    glBindTexture(GL_TEXTURE_3D, 0);
    return t;
}

void FluidApp::createGridTextures()
{
    texAVelX_     = makeGridTex(GL_R32I);
    texAVelY_     = makeGridTex(GL_R32I);
    texAVelZ_     = makeGridTex(GL_R32I);
    texAWeight_   = makeGridTex(GL_R32I);

    texVelX_      = makeGridTex(GL_R32F);
    texVelY_      = makeGridTex(GL_R32F);
    texVelZ_      = makeGridTex(GL_R32F);

    texVelXSave_  = makeGridTex(GL_R32F);
    texVelYSave_  = makeGridTex(GL_R32F);
    texVelZSave_  = makeGridTex(GL_R32F);

    texWeightF_   = makeGridTex(GL_R32F);
    texPressureA_ = makeGridTex(GL_R32F);
    texPressureB_ = makeGridTex(GL_R32F);
    texDivergence_= makeGridTex(GL_R32F);
    texCellType_  = makeGridTex(GL_R8UI);
}

// -----------------------------------------------------------------------
// Seed PARTICLE_COUNT particles uniformly inside the annular pool.
// -----------------------------------------------------------------------
void FluidApp::initParticleBuffer()
{
    std::vector<ParticleCpu> particles(PARTICLE_COUNT);
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    const float cx = 63.5f, cz = 63.5f;
    const float rIn = 16.0f, rOut = 58.0f;
    const float yLo = 9.0f,  yHi  = 44.0f;   // tall initial column (~35 voxels)

    // Bowl pool: alive, distributed in the annulus.
    for (int i = 0; i < POOL_BASE_COUNT; ++i) {
        float r = std::sqrt(u01(rng) * (rOut*rOut - rIn*rIn) + rIn*rIn);
        float th = u01(rng) * 6.2831853f;
        ParticleCpu& p = particles[i];
        p.pos[0] = cx + r * std::cos(th) + (u01(rng) - 0.5f) * 0.6f;
        p.pos[1] = yLo + u01(rng) * (yHi - yLo);
        p.pos[2] = cz + r * std::sin(th) + (u01(rng) - 0.5f) * 0.6f;
        p.life   = 1.0f;
        p.vel[0] = p.vel[1] = p.vel[2] = 0.0f;
        p.pad    = 0.0f;
    }
    // Pinch reserve: dead, parked at the origin until a pinch spawns them.
    for (int i = POOL_BASE_COUNT; i < PARTICLE_COUNT; ++i) {
        ParticleCpu& p = particles[i];
        p.pos[0] = p.pos[1] = p.pos[2] = 0.0f;
        p.life   = -1.0f;
        p.vel[0] = p.vel[1] = p.vel[2] = 0.0f;
        p.pad    = 0.0f;
    }

    glGenBuffers(1, &particleSSBO_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleSSBO_);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 particles.size() * sizeof(ParticleCpu),
                 particles.data(),
                 GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// -----------------------------------------------------------------------
void FluidApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    bool ok = true;
    ok &= loadProgram("shaders/flip/clear_grid.glsl",       progClear_);
    ok &= loadProgram("shaders/flip/particle_recycle.glsl", progRecycle_);
    ok &= loadProgram("shaders/flip/p2g.glsl",              progP2G_);
    ok &= loadProgram("shaders/flip/normalize_vel.glsl",    progNormalize_);
    ok &= loadProgram("shaders/flip/snapshot_vel.glsl",     progSnapshot_);
    ok &= loadProgram("shaders/flip/mark_cells.glsl",       progMark_);
    ok &= loadProgram("shaders/flip/divergence.glsl",       progDivergence_);
    ok &= loadProgram("shaders/flip/pressure_jacobi.glsl",  progJacobi_);
    ok &= loadProgram("shaders/flip/subtract_grad.glsl",    progSubGrad_);
    ok &= loadProgram("shaders/flip/g2p_advect.glsl",       progG2P_);
    ok &= loadProgram("shaders/flip/voxel_splat.glsl",      progSplat_);
    if (!ok) {
        fprintf(stderr, "FluidApp: shader compilation failed; sim disabled\n");
        return;
    }

    uRecycleCount_       = glGetUniformLocation(progRecycle_, "uParticleCount");
    uRecycleBase_        = glGetUniformLocation(progRecycle_, "uBaseCount");
    uRecycleFrame_       = glGetUniformLocation(progRecycle_, "uFrame");
    uRecycleFill_        = glGetUniformLocation(progRecycle_, "uFillHeight");
    uRecyclePinchActive_ = glGetUniformLocation(progRecycle_, "uPinchActive");
    uRecyclePinchPos_    = glGetUniformLocation(progRecycle_, "uPinchPos");
    uRecyclePinchStart_  = glGetUniformLocation(progRecycle_, "uPinchSpawnStart");
    uRecyclePinchCount_  = glGetUniformLocation(progRecycle_, "uPinchSpawnCount");
    uP2GCount_     = glGetUniformLocation(progP2G_,       "uParticleCount");
    uSnapGravity_  = glGetUniformLocation(progSnapshot_,  "uGravity");
    uSnapDt_       = glGetUniformLocation(progSnapshot_,  "uDt");
    uG2PCount_     = glGetUniformLocation(progG2P_,       "uParticleCount");
    uG2PDt_        = glGetUniformLocation(progG2P_,       "uDt");
    uG2PAlpha_     = glGetUniformLocation(progG2P_,       "uFlipAlpha");
    uSplatCount_   = glGetUniformLocation(progSplat_,     "uParticleCount");
    uSplatDt_      = glGetUniformLocation(progSplat_,     "uDt");

    createGridTextures();
    initParticleBuffer();
    firstFrame_ = true;
    frameCounter_ = 0;
}

// -----------------------------------------------------------------------
// Read finger landmark, smooth, store. Tilt vector is computed in draw().
// -----------------------------------------------------------------------
void FluidApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    // ---- ENV var overrides for headless testing ---------------------------
    // TILT_FAKE_X / TILT_FAKE_Z (range [-1,1]) drive the gravity tilt vector.
    // PINCH_FAKE_X / PINCH_FAKE_Z (range [0,1], image-space) enable the pinch
    // droplet stream at that location. Either or both may be set.
    const char* envTiltX  = std::getenv("TILT_FAKE_X");
    const char* envTiltZ  = std::getenv("TILT_FAKE_Z");
    const char* envPinchX = std::getenv("PINCH_FAKE_X");
    const char* envPinchZ = std::getenv("PINCH_FAKE_Z");
    if (envTiltX || envTiltZ || envPinchX || envPinchZ) {
        fingerActive_ = true;
        float tx = envTiltX ? clampf(std::atof(envTiltX), -1.0f, 1.0f) : 0.0f;
        float tz = envTiltZ ? clampf(std::atof(envTiltZ), -1.0f, 1.0f) : 0.0f;
        fingerSmoothX_ = 0.5f + 0.5f * tx;
        fingerSmoothY_ = 0.5f + 0.5f * tz;

        if (envPinchX || envPinchZ) {
            float px = envPinchX ? clampf(std::atof(envPinchX), 0.0f, 1.0f) : 0.5f;
            float pz = envPinchZ ? clampf(std::atof(envPinchZ), 0.0f, 1.0f) : 0.5f;
            setPinch(px, pz);
        } else {
            pinchActive_ = false;
        }
        return;
    }

    fingerActive_ = hand.hand_detected;
    if (!hand.hand_detected) {
        pinchActive_ = false;
        return;
    }

    float rawX = clampf(hand.lm_x[8], 0.0f, 1.0f);
    float rawY = clampf(hand.lm_y[8], 0.0f, 1.0f);

    // Low-pass to kill landmark jitter without smearing slosh.
    fingerSmoothX_ += 0.25f * (rawX - fingerSmoothX_);
    fingerSmoothY_ += 0.25f * (rawY - fingerSmoothY_);

    // Pinch detection: thumb tip (4) ↔ index tip (8) Euclidean distance.
    float pinchDist = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                                 hand.lm_y[4] - hand.lm_y[8]);
    if (pinchDist < PINCH_THRESHOLD) {
        float mxn = 0.5f * (hand.lm_x[4] + hand.lm_x[8]);
        float mzn = 0.5f * (hand.lm_y[4] + hand.lm_y[8]);
        setPinch(mxn, mzn);
    } else {
        pinchActive_ = false;
    }
}

// Map a normalised (image-space) pinch midpoint to a voxel-space spawn point,
// pushing it outside the bowl's inner dead zone if necessary.
void FluidApp::setPinch(float nx, float nz)
{
    float x = clampf(nx * 128.0f, 16.0f, 112.0f);
    float z = clampf(nz * 128.0f, 16.0f, 112.0f);
    const float CX = 63.5f, CZ = 63.5f;
    const float SAFE_R = 20.0f;  // outside R_INNER (14.5) with margin
    float dx = x - CX, dz = z - CZ;
    float d  = std::hypot(dx, dz);
    if (d < SAFE_R) {
        if (d < 1e-4f) { dx = SAFE_R; dz = 0.0f; }
        else           { float s = SAFE_R / d; dx *= s; dz *= s; }
        x = CX + dx;
        z = CZ + dz;
    }
    pinchVoxelX_ = x;
    pinchVoxelZ_ = z;
    pinchVoxelY_ = PINCH_SPAWN_Y;
    pinchActive_ = true;
}

// -----------------------------------------------------------------------
// Image-binding helpers — keep dispatch sites readable.
// -----------------------------------------------------------------------
void FluidApp::bindGridImages_Clear()
{
    glBindImageTexture(0, texAVelX_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32I);
    glBindImageTexture(1, texAVelY_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32I);
    glBindImageTexture(2, texAVelZ_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32I);
    glBindImageTexture(3, texAWeight_,  0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32I);
    glBindImageTexture(4, texVelX_,     0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(5, texVelY_,     0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(6, texVelZ_,     0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(7, texCellType_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8UI);
    // Pressure textures zeroed via glClearTexImage (avoids exceeding 8-unit limit).
}

void FluidApp::bindGridImages_P2G()
{
    glBindImageTexture(1, texAVelX_,   0, GL_TRUE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(2, texAVelY_,   0, GL_TRUE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(3, texAVelZ_,   0, GL_TRUE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(4, texAWeight_, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32I);
}

void FluidApp::bindGridImages_Normalize()
{
    glBindImageTexture(0, texAVelX_,   0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32I);
    glBindImageTexture(1, texAVelY_,   0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32I);
    glBindImageTexture(2, texAVelZ_,   0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32I);
    glBindImageTexture(3, texAWeight_, 0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32I);
    glBindImageTexture(4, texVelX_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(5, texVelY_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(6, texVelZ_,    0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(7, texWeightF_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
}

void FluidApp::bindGridImages_Snapshot()
{
    glBindImageTexture(0, texVelX_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, texVelY_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(2, texVelZ_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(3, texVelXSave_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(4, texVelYSave_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(5, texVelZSave_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(6, texWeightF_,  0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
}

void FluidApp::bindGridImages_MarkCells()
{
    glBindImageTexture(0, texWeightF_,  0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(1, texCellType_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8UI);
}

void FluidApp::bindGridImages_Divergence()
{
    glBindImageTexture(0, texVelX_,       0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(1, texVelY_,       0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(2, texVelZ_,       0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(3, texCellType_,   0, GL_TRUE, 0, GL_READ_ONLY,  GL_R8UI);
    glBindImageTexture(4, texDivergence_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
}

void FluidApp::bindGridImages_Jacobi(GLuint pIn, GLuint pOut)
{
    glBindImageTexture(0, pIn,             0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(1, pOut,            0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(2, texDivergence_,  0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(3, texCellType_,    0, GL_TRUE, 0, GL_READ_ONLY,  GL_R8UI);
}

void FluidApp::bindGridImages_SubtractGrad(GLuint pSource)
{
    glBindImageTexture(0, texVelX_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, texVelY_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(2, texVelZ_,     0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(3, pSource,      0, GL_TRUE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(4, texCellType_, 0, GL_TRUE, 0, GL_READ_ONLY,  GL_R8UI);
}

void FluidApp::bindGridImages_G2P()
{
    glBindImageTexture(0, texWeightF_,  0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, texVelX_,     0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, texVelY_,     0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, texVelZ_,     0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(4, texVelXSave_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(5, texVelYSave_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(6, texVelZSave_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
}

void FluidApp::bindParticleSSBO()
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleSSBO_);
}

void FluidApp::barrier()
{
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_SHADER_STORAGE_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);
}

void FluidApp::dispatchParticles()
{
    GLuint groups = (PARTICLE_COUNT + 255) / 256;
    glDispatchCompute(groups, 1, 1);
}

void FluidApp::dispatchGrid()
{
    glDispatchCompute(GRID_W / 4, GRID_H / 4, GRID_D / 4);
}

// -----------------------------------------------------------------------
// One simulation step + voxel splat.
// -----------------------------------------------------------------------
void FluidApp::draw(Renderer& renderer)
{
    if (!progClear_) return;

    // Wall-clock dt, clamped.
    auto now = std::chrono::steady_clock::now();
    float dt;
    if (firstFrame_) {
        dt = 1.0f / 60.0f;
        firstFrame_ = false;
    } else {
        dt = std::chrono::duration<float>(now - lastTick_).count();
    }
    lastTick_ = now;
    dt = clampf(dt, DT_MIN, DT_MAX);
    frameCounter_++;

    // --- Tilt vector from finger position (centered, with low-pass already
    //     applied in update()). When no hand visible, gravity points straight
    //     down so the pool relaxes.
    float tiltX = 0.0f, tiltZ = 0.0f;
    if (fingerActive_) {
        // MediaPipe Y maps to world Z (camera image vertical → table depth).
        tiltX = (fingerSmoothX_ - 0.5f) * 2.0f;   // [-1, 1]
        tiltZ = (fingerSmoothY_ - 0.5f) * 2.0f;
    }
    float gx =  TILT_K * tiltX;
    float gy = -1.0f;
    float gz =  TILT_K * tiltZ;
    float gn = std::sqrt(gx*gx + gy*gy + gz*gz);
    gx = gx / gn * GRAVITY_MAG;
    gy = gy / gn * GRAVITY_MAG;
    gz = gz / gn * GRAVITY_MAG;

    bindParticleSSBO();

    // ----- A. Recycle dead particles (no barrier needed before, untouched) -
    glUseProgram(progRecycle_);
    glUniform1ui(uRecycleCount_, (GLuint)PARTICLE_COUNT);
    glUniform1ui(uRecycleBase_,  (GLuint)POOL_BASE_COUNT);
    glUniform1ui(uRecycleFrame_, (GLuint)frameCounter_);
    glUniform1f (uRecycleFill_,  35.0f);

    GLuint pinchSpawnCount = pinchActive_ ? (GLuint)PINCH_SPAWN_PER_FRAME : 0u;
    glUniform1ui(uRecyclePinchActive_, pinchSpawnCount > 0u ? 1u : 0u);
    glUniform3f (uRecyclePinchPos_, pinchVoxelX_, pinchVoxelY_, pinchVoxelZ_);
    glUniform1ui(uRecyclePinchStart_, (GLuint)pinchSpawnCursor_);
    glUniform1ui(uRecyclePinchCount_, pinchSpawnCount);

    if (pinchSpawnCount > 0u) {
        const GLuint poolPinchSize = (GLuint)(PARTICLE_COUNT - POOL_BASE_COUNT);
        pinchSpawnCursor_ = (pinchSpawnCursor_ + pinchSpawnCount) % poolPinchSize;
    }

    dispatchParticles();
    barrier();

    // ----- B. Clear grid ---------------------------------------------------
    // Zero pressure textures on the CPU to avoid exceeding 8 image unit limit.
    const float kZero = 0.0f;
    glClearTexImage(texPressureA_, 0, GL_RED, GL_FLOAT, &kZero);
    glClearTexImage(texPressureB_, 0, GL_RED, GL_FLOAT, &kZero);
    glUseProgram(progClear_);
    bindGridImages_Clear();
    dispatchGrid();
    barrier();

    // ----- C. P2G scatter --------------------------------------------------
    glUseProgram(progP2G_);
    bindGridImages_P2G();
    glUniform1ui(uP2GCount_, (GLuint)PARTICLE_COUNT);
    dispatchParticles();
    barrier();

    // ----- D. Decode + normalize (no gravity yet) -------------------------
    glUseProgram(progNormalize_);
    bindGridImages_Normalize();
    dispatchGrid();
    barrier();

    // ----- D2. Snapshot pre-gravity velocity, then apply gravity ---------
    // The snapshot must precede gravity so the G2P FLIP delta
    // (vNew − vSave) carries both gravity and pressure correction.
    glUseProgram(progSnapshot_);
    bindGridImages_Snapshot();
    glUniform3f(uSnapGravity_, gx, gy, gz);
    glUniform1f(uSnapDt_, dt);
    dispatchGrid();
    barrier();

    // ----- E. Mark cell types ---------------------------------------------
    glUseProgram(progMark_);
    bindGridImages_MarkCells();
    dispatchGrid();
    barrier();

    // ----- F. Divergence ---------------------------------------------------
    glUseProgram(progDivergence_);
    bindGridImages_Divergence();
    dispatchGrid();
    barrier();

    // ----- G. Pressure Jacobi (ping-pong) ---------------------------------
    glUseProgram(progJacobi_);
    GLuint pIn  = texPressureA_;
    GLuint pOut = texPressureB_;
    for (int i = 0; i < PRESSURE_ITERATIONS; ++i) {
        bindGridImages_Jacobi(pIn, pOut);
        dispatchGrid();
        barrier();
        GLuint tmp = pIn; pIn = pOut; pOut = tmp;
    }
    GLuint pFinal = pIn; // last write landed in `pIn` after the swap

    // ----- H. Subtract pressure gradient ----------------------------------
    glUseProgram(progSubGrad_);
    bindGridImages_SubtractGrad(pFinal);
    dispatchGrid();
    barrier();

    // ----- I. G2P + advect + collide --------------------------------------
    glUseProgram(progG2P_);
    bindGridImages_G2P();
    glUniform1ui(uG2PCount_, (GLuint)PARTICLE_COUNT);
    glUniform1f (uG2PDt_,    dt);
    glUniform1f (uG2PAlpha_, FLIP_ALPHA);
    dispatchParticles();
    barrier();

    // ----- J. Splat live particles into 128×64×128 display volume ---------
    glUseProgram(progSplat_);
    glBindImageTexture(1, renderer.getVoxelTextureID(),
                       0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform1ui(uSplatCount_, (GLuint)PARTICLE_COUNT);
    glUniform1f (uSplatDt_,    dt);
    dispatchParticles();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
}

// -----------------------------------------------------------------------
void FluidApp::teardown(Renderer& /*renderer*/)
{
    GLuint progs[] = {progClear_, progRecycle_, progP2G_, progNormalize_,
                      progSnapshot_, progMark_, progDivergence_, progJacobi_,
                      progSubGrad_, progG2P_, progSplat_};
    for (GLuint p : progs) if (p) glDeleteProgram(p);
    progClear_ = progRecycle_ = progP2G_ = progNormalize_ = progSnapshot_ = 0;
    progMark_ = progDivergence_ = progJacobi_ = progSubGrad_ = 0;
    progG2P_  = progSplat_ = 0;

    GLuint texs[] = {texAVelX_, texAVelY_, texAVelZ_, texAWeight_,
                     texVelX_, texVelY_, texVelZ_,
                     texVelXSave_, texVelYSave_, texVelZSave_,
                     texWeightF_, texPressureA_, texPressureB_,
                     texDivergence_, texCellType_};
    glDeleteTextures(sizeof(texs)/sizeof(texs[0]), texs);
    texAVelX_ = texAVelY_ = texAVelZ_ = texAWeight_ = 0;
    texVelX_ = texVelY_ = texVelZ_ = 0;
    texVelXSave_ = texVelYSave_ = texVelZSave_ = 0;
    texWeightF_ = texPressureA_ = texPressureB_ = 0;
    texDivergence_ = texCellType_ = 0;

    if (particleSSBO_) { glDeleteBuffers(1, &particleSSBO_); particleSSBO_ = 0; }
}
