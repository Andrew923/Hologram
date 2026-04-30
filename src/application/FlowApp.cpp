#include "FlowApp.h"
#include "../engine/Renderer.h"
#include <epoxy/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Particle layout matches GLSL `Particle { vec4 pos; }` (16 bytes).
struct ParticleCpu {
    float pos[3];
    float age;
};
static_assert(sizeof(ParticleCpu) == 16, "ParticleCpu layout mismatch");

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

bool FlowApp::loadProgram(const char* path, GLuint& outProg)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FlowApp: cannot open shader: %s\n", path);
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
        fprintf(stderr, "FlowApp: shader compile error (%s):\n%s\n", path, log);
        glDeleteShader(cs);
        return false;
    }

    outProg = glCreateProgram();
    glAttachShader(outProg, cs);
    glLinkProgram(outProg);
    glDeleteShader(cs);

    glGetProgramiv(outProg, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(outProg, sizeof(log), nullptr, log);
        fprintf(stderr, "FlowApp: shader link error (%s):\n%s\n", path, log);
        glDeleteProgram(outProg);
        outProg = 0;
        return false;
    }
    return true;
}

// Per-attractor visualization params: where to centre/scale the world
// coordinates so the attractor fits the voxel volume, and the integration
// step + substep count for stability.
void FlowApp::getTransform(int idx, float origin[3], float scale[3],
                           float& dt, int& substeps, float color[4]) const
{
    switch (idx) {
        case 0: // Lorenz: roughly x∈±25, y∈±30, z∈[0,55]
            origin[0] = 0.0f;  origin[1] = 0.0f;  origin[2] = 25.0f;
            scale[0]  = 1.6f;  scale[1]  = 0.8f;  scale[2]  = 1.6f;
            dt = 0.005f;  substeps = 6;
            color[0]=1.0f; color[1]=1.0f; color[2]=1.0f; color[3]=1.0f; // white
            break;
        case 1: // Aizawa: roughly x∈±2, y∈±2, z∈[-1,1.6]
            origin[0] = 0.0f;  origin[1] = 0.0f;  origin[2] = 0.3f;
            scale[0]  = 22.0f; scale[1]  = 14.0f; scale[2]  = 28.0f;
            dt = 0.01f;   substeps = 4;
            color[0]=0.0f; color[1]=1.0f; color[2]=1.0f; color[3]=1.0f; // cyan
            break;
        case 2: // Halvorsen: roughly ±15
            origin[0] = -5.0f; origin[1] = -5.0f; origin[2] = -5.0f;
            scale[0]  = 2.0f;  scale[1]  = 1.0f;  scale[2]  = 2.0f;
            dt = 0.005f;  substeps = 4;
            color[0]=1.0f; color[1]=0.0f; color[2]=1.0f; color[3]=1.0f; // magenta
            break;
        default: // Thomas: roughly ±5
            origin[0] = 0.0f;  origin[1] = 0.0f;  origin[2] = 0.0f;
            scale[0]  = 8.0f;  scale[1]  = 4.5f;  scale[2]  = 8.0f;
            dt = 0.05f;   substeps = 4;
            color[0]=1.0f; color[1]=1.0f; color[2]=0.0f; color[3]=1.0f; // yellow
            break;
    }
}

// Seed particles randomly inside the attractor's basin so they start as a
// shapeless cloud and converge onto the attractor over the first second.
void FlowApp::initParticles(int attractorIdx)
{
    std::vector<ParticleCpu> particles(PARTICLE_COUNT);
    std::mt19937 rng(0xBADC0DE);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float cx, cy, cz, rx, ry, rz;
    switch (attractorIdx) {
        case 0:  cx= 0.0f; cy= 0.0f; cz=25.0f; rx=15.0f; ry=15.0f; rz=15.0f; break;
        case 1:  cx= 0.0f; cy= 0.0f; cz= 0.5f; rx= 1.5f; ry= 1.5f; rz= 1.0f; break;
        case 2:  cx=-5.0f; cy=-5.0f; cz=-5.0f; rx=10.0f; ry=10.0f; rz=10.0f; break;
        default: cx= 0.0f; cy= 0.0f; cz= 0.0f; rx= 3.0f; ry= 3.0f; rz= 3.0f; break;
    }

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        ParticleCpu& p = particles[i];
        p.pos[0] = cx + (u01(rng) - 0.5f) * 2.0f * rx;
        p.pos[1] = cy + (u01(rng) - 0.5f) * 2.0f * ry;
        p.pos[2] = cz + (u01(rng) - 0.5f) * 2.0f * rz;
        p.age    = 0.0f;
    }

    if (!particleSSBO_) glGenBuffers(1, &particleSSBO_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleSSBO_);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 particles.size() * sizeof(ParticleCpu),
                 particles.data(),
                 GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void FlowApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    bool ok = true;
    ok &= loadProgram("shaders/flow/step.glsl",  progStep_);
    ok &= loadProgram("shaders/flow/splat.glsl", progSplat_);
    if (!ok) {
        fprintf(stderr, "FlowApp: shader load failed; app disabled\n");
        return;
    }

    uStepCount_     = glGetUniformLocation(progStep_,  "uCount");
    uStepDt_        = glGetUniformLocation(progStep_,  "uDt");
    uStepSubsteps_  = glGetUniformLocation(progStep_,  "uSubsteps");
    uStepAttractor_ = glGetUniformLocation(progStep_,  "uAttractor");
    uSplatCount_    = glGetUniformLocation(progSplat_, "uCount");
    uSplatOrigin_   = glGetUniformLocation(progSplat_, "uOrigin");
    uSplatScale_    = glGetUniformLocation(progSplat_, "uScale");
    uSplatColor_    = glGetUniformLocation(progSplat_, "uColor");

    attractorIdx_ = 0;
    initParticles(attractorIdx_);
    pinchPrev_       = false;
    pinchHoldFrames_ = 0;
}

void FlowApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    fingerActive_ = hand.hand_detected;

    bool pinching = false;
    if (hand.hand_detected) {
        float pinchDist = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                                     hand.lm_y[4] - hand.lm_y[8]);
        pinching = (pinchDist < 0.06f);
        fingerSmoothX_ += 0.25f * (clampf(hand.lm_x[8], 0.0f, 1.0f) - fingerSmoothX_);
        fingerSmoothY_ += 0.25f * (clampf(hand.lm_y[8], 0.0f, 1.0f) - fingerSmoothY_);
    }

    // Edge-triggered pinch: rising edge advances attractor.
    if (pinching && !pinchPrev_) {
        attractorIdx_ = (attractorIdx_ + 1) % NUM_ATTRACTORS;
        initParticles(attractorIdx_);
    }
    pinchPrev_ = pinching;
}

void FlowApp::draw(Renderer& renderer)
{
    if (!progStep_ || !progSplat_) return;

    float origin[3], scale[3], color[4];
    float dt;
    int substeps;
    getTransform(attractorIdx_, origin, scale, dt, substeps, color);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleSSBO_);

    // Step the ODE.
    glUseProgram(progStep_);
    glUniform1ui(uStepCount_,     (GLuint)PARTICLE_COUNT);
    glUniform1f (uStepDt_,        dt);
    glUniform1i (uStepSubsteps_,  substeps);
    glUniform1i (uStepAttractor_, attractorIdx_);
    GLuint groups = (PARTICLE_COUNT + 255) / 256;
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Splat into the voxel volume.
    glUseProgram(progSplat_);
    glBindImageTexture(1, renderer.getVoxelTextureID(), 0, GL_TRUE, 0,
                       GL_WRITE_ONLY, GL_RGBA8);
    glUniform1ui(uSplatCount_,  (GLuint)PARTICLE_COUNT);
    glUniform3f (uSplatOrigin_, origin[0], origin[1], origin[2]);
    glUniform3f (uSplatScale_,  scale[0],  scale[1],  scale[2]);
    glUniform4f (uSplatColor_,  color[0],  color[1],  color[2], color[3]);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
}

void FlowApp::teardown(Renderer& /*renderer*/)
{
    if (progStep_)     { glDeleteProgram(progStep_);  progStep_  = 0; }
    if (progSplat_)    { glDeleteProgram(progSplat_); progSplat_ = 0; }
    if (particleSSBO_) { glDeleteBuffers(1, &particleSSBO_); particleSSBO_ = 0; }
}
