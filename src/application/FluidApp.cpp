#include "FluidApp.h"
#include "../engine/Renderer.h"
#include <epoxy/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr float REST_HEIGHT   = 8.0f;
static constexpr int   HEIGHT_MAP_W  = 128;
static constexpr int   HEIGHT_MAP_H  = 128;
static constexpr float PINCH_MIN_DIST = 0.03f;
static constexpr float PINCH_RANGE    = 0.22f;

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// -----------------------------------------------------------------------
// Load and compile a compute shader from file
// -----------------------------------------------------------------------
bool FluidApp::loadShader(const char* path)
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
        fprintf(stderr, "FluidApp: shader compile error:\n%s\n", log);
        glDeleteShader(cs);
        return false;
    }

    fluidProg_ = glCreateProgram();
    glAttachShader(fluidProg_, cs);
    glLinkProgram(fluidProg_);
    glDeleteShader(cs);

    glGetProgramiv(fluidProg_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(fluidProg_, sizeof(log), nullptr, log);
        fprintf(stderr, "FluidApp: shader link error:\n%s\n", log);
        glDeleteProgram(fluidProg_);
        fluidProg_ = 0;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------
void FluidApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    if (!loadShader("shaders/fluid_sim.glsl")) return;

    uFingerXZLoc_     = glGetUniformLocation(fluidProg_, "uFingerXZ");
    uImpulseLoc_      = glGetUniformLocation(fluidProg_, "uImpulse");
    uFingerActiveLoc_ = glGetUniformLocation(fluidProg_, "uFingerActive");

    // Initialise height maps to REST_HEIGHT
    std::vector<float> initData(HEIGHT_MAP_W * HEIGHT_MAP_H, REST_HEIGHT);

    glGenTextures(3, hTex_);
    for (int i = 0; i < 3; ++i) {
        glBindTexture(GL_TEXTURE_2D, hTex_[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                     HEIGHT_MAP_W, HEIGHT_MAP_H, 0,
                     GL_RED, GL_FLOAT, initData.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    hCurIdx_  = 0;
    hPrevIdx_ = 1;
    hNextIdx_ = 2;
}

// -----------------------------------------------------------------------
// Update — CPU side: read hand, compute finger position and impulse
// -----------------------------------------------------------------------
void FluidApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    fingerActive_ = hand.hand_detected;
    if (!hand.hand_detected) return;

    fingerX_ = clampf(hand.lm_x[8] * (float)HEIGHT_MAP_W, 16.0f, 112.0f);
    fingerZ_ = clampf(hand.lm_y[8] * (float)HEIGHT_MAP_H, 16.0f, 112.0f);

    float pinchDist = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
    float pinchT = clampf((pinchDist - PINCH_MIN_DIST) / PINCH_RANGE, 0.0f, 1.0f);
    impulse_ = 0.3f + pinchT * 3.7f;
}

// -----------------------------------------------------------------------
// Draw — GPU dispatch; writes directly to voxel texture
// -----------------------------------------------------------------------
void FluidApp::draw(Renderer& renderer)
{
    if (!fluidProg_) return;

    glUseProgram(fluidProg_);

    if (uFingerXZLoc_     >= 0) glUniform2f(uFingerXZLoc_, fingerX_, fingerZ_);
    if (uImpulseLoc_      >= 0) glUniform1f(uImpulseLoc_,  impulse_);
    if (uFingerActiveLoc_ >= 0) glUniform1i(uFingerActiveLoc_, fingerActive_ ? 1 : 0);

    // Voxel grid — write only (binding 0)
    glBindImageTexture(0, renderer.getVoxelTextureID(),
                       0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
    // Height maps (bindings 1–3)
    glBindImageTexture(1, hTex_[hCurIdx_],  0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(2, hTex_[hPrevIdx_], 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(3, hTex_[hNextIdx_], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // 128×128 grid, workgroup 8×8 → 16×16 groups
    glDispatchCompute(16, 16, 1);

    // Ensure writes are visible to the slicer's texture sampler
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);

    // Ping-pong: hPrev←hCur, hCur←hNext, hNext←old hPrev
    int tmp  = hPrevIdx_;
    hPrevIdx_ = hCurIdx_;
    hCurIdx_  = hNextIdx_;
    hNextIdx_ = tmp;
}

// -----------------------------------------------------------------------
// Teardown
// -----------------------------------------------------------------------
void FluidApp::teardown(Renderer& /*renderer*/)
{
    if (fluidProg_) { glDeleteProgram(fluidProg_); fluidProg_ = 0; }
    glDeleteTextures(3, hTex_);
    hTex_[0] = hTex_[1] = hTex_[2] = 0;
}
