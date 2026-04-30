#include "WaveApp.h"
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

bool WaveApp::loadShader(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "WaveApp: cannot open shader: %s\n", path);
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
        fprintf(stderr, "WaveApp: shader compile error:\n%s\n", log);
        glDeleteShader(cs);
        return false;
    }

    waveProg_ = glCreateProgram();
    glAttachShader(waveProg_, cs);
    glLinkProgram(waveProg_);
    glDeleteShader(cs);

    glGetProgramiv(waveProg_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(waveProg_, sizeof(log), nullptr, log);
        fprintf(stderr, "WaveApp: shader link error:\n%s\n", log);
        glDeleteProgram(waveProg_);
        waveProg_ = 0;
        return false;
    }
    return true;
}

void WaveApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    if (!loadShader("shaders/fluid_sim.glsl")) return;

    uFingerXZLoc_     = glGetUniformLocation(waveProg_, "uFingerXZ");
    uImpulseLoc_      = glGetUniformLocation(waveProg_, "uImpulse");
    uFingerActiveLoc_ = glGetUniformLocation(waveProg_, "uFingerActive");

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

void WaveApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    fingerActive_ = hand.hand_detected;
    if (!hand.hand_detected) return;

    fingerX_ = clampf(hand.lm_x[8] * (float)HEIGHT_MAP_W, 16.0f, 112.0f);
    fingerZ_ = clampf(hand.lm_y[8] * (float)HEIGHT_MAP_H, 16.0f, 112.0f);

    float pinchDist = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
    float pinchT = clampf((pinchDist - PINCH_MIN_DIST) / PINCH_RANGE, 0.0f, 1.0f);
    // Impulse range scales with the shader's MAX_H (now 48 vox); cap raised
    // to 12 so an open hand can drive a wave to the new ceiling.
    impulse_ = 0.3f + pinchT * 11.7f;
}

void WaveApp::draw(Renderer& renderer)
{
    if (!waveProg_) return;

    glUseProgram(waveProg_);

    if (uFingerXZLoc_     >= 0) glUniform2f(uFingerXZLoc_, fingerX_, fingerZ_);
    if (uImpulseLoc_      >= 0) glUniform1f(uImpulseLoc_,  impulse_);
    if (uFingerActiveLoc_ >= 0) glUniform1i(uFingerActiveLoc_, fingerActive_ ? 1 : 0);

    glBindImageTexture(0, renderer.getVoxelTextureID(),
                       0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glBindImageTexture(1, hTex_[hCurIdx_],  0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(2, hTex_[hPrevIdx_], 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
    glBindImageTexture(3, hTex_[hNextIdx_], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    glDispatchCompute(16, 16, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);

    int tmp  = hPrevIdx_;
    hPrevIdx_ = hCurIdx_;
    hCurIdx_  = hNextIdx_;
    hNextIdx_ = tmp;
}

void WaveApp::teardown(Renderer& /*renderer*/)
{
    if (waveProg_) { glDeleteProgram(waveProg_); waveProg_ = 0; }
    glDeleteTextures(3, hTex_);
    hTex_[0] = hTex_[1] = hTex_[2] = 0;
}
