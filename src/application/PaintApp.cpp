#include "PaintApp.h"
#include "../engine/Renderer.h"
#include <epoxy/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

bool PaintApp::loadProgram(const char* path, GLuint& outProg)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "PaintApp: cannot open shader: %s\n", path);
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
        fprintf(stderr, "PaintApp: shader compile error (%s):\n%s\n", path, log);
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
        fprintf(stderr, "PaintApp: shader link error (%s):\n%s\n", path, log);
        glDeleteProgram(outProg);
        outProg = 0;
        return false;
    }
    return true;
}

void PaintApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    if (!loadProgram("shaders/paint/step.glsl", progStep_)) {
        fprintf(stderr, "PaintApp: shader load failed; app disabled\n");
        return;
    }

    uCursorLoc_       = glGetUniformLocation(progStep_, "uCursor");
    uCursorRadiusLoc_ = glGetUniformLocation(progStep_, "uCursorRadius");
    uCursorActiveLoc_ = glGetUniformLocation(progStep_, "uCursorActive");
    uDecayLoc_        = glGetUniformLocation(progStep_, "uDecay");

    glGenTextures(1, &texAge_);
    glBindTexture(GL_TEXTURE_3D, texAge_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R8, 128, 64, 128);
    glBindTexture(GL_TEXTURE_3D, 0);

    const float zero = 0.0f;
    glClearTexImage(texAge_, 0, GL_RED, GL_FLOAT, &zero);

    fingerActive_ = false;
}

void PaintApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    fingerActive_ = hand.hand_detected;
    if (!hand.hand_detected) return;

    float lx = clampf(hand.lm_x[8], 0.0f, 1.0f);
    float ly = clampf(hand.lm_y[8], 0.0f, 1.0f);

    cursorX_ = clampf(lx * 128.0f, 8.0f, 120.0f);
    cursorZ_ = clampf(ly * 128.0f, 8.0f, 120.0f);

    // Pinch openness drives Y. Tight pinch (~0.03) = low; wide (>0.25) = high.
    float pinchDist = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                                 hand.lm_y[4] - hand.lm_y[8]);
    float t = clampf((pinchDist - 0.03f) / 0.22f, 0.0f, 1.0f);
    cursorY_ = clampf(12.0f + t * 44.0f, 12.0f, 56.0f);
}

void PaintApp::draw(Renderer& renderer)
{
    if (!progStep_) return;

    glUseProgram(progStep_);
    glBindImageTexture(0, texAge_, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8);
    glBindImageTexture(1, renderer.getVoxelTextureID(), 0, GL_TRUE, 0,
                       GL_WRITE_ONLY, GL_RGBA8);

    glUniform3f(uCursorLoc_,        cursorX_, cursorY_, cursorZ_);
    glUniform1f(uCursorRadiusLoc_,  1.5f);
    glUniform1f(uCursorActiveLoc_,  fingerActive_ ? 1.0f : 0.0f);
    glUniform1f(uDecayLoc_,         0.012f);   // ~85 frames to fade out

    glDispatchCompute(128 / 4, 64 / 4, 128 / 4);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
}

void PaintApp::teardown(Renderer& /*renderer*/)
{
    if (progStep_) { glDeleteProgram(progStep_); progStep_ = 0; }
    if (texAge_)   { glDeleteTextures(1, &texAge_); texAge_ = 0; }
}
