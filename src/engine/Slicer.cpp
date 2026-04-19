#include "Slicer.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Slicer::~Slicer() { shutdown(); }

// -----------------------------------------------------------------------
// Load compute shader from file, compile, link
// -----------------------------------------------------------------------
bool Slicer::loadComputeShader(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Slicer: cannot open shader: %s\n", path);
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
        char log[1024];
        glGetShaderInfoLog(cs, sizeof(log), nullptr, log);
        fprintf(stderr, "Slicer: compute shader compile error:\n%s\n", log);
        glDeleteShader(cs);
        return false;
    }

    computeProg_ = glCreateProgram();
    glAttachShader(computeProg_, cs);
    glLinkProgram(computeProg_);
    glDeleteShader(cs);

    glGetProgramiv(computeProg_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(computeProg_, sizeof(log), nullptr, log);
        fprintf(stderr, "Slicer: compute shader link error:\n%s\n", log);
        glDeleteProgram(computeProg_);
        computeProg_ = 0;
        return false;
    }

    return true;
}

void Slicer::cacheUniformLocations()
{
    if (!computeProg_) return;
    uPanelOffsetLoc_ = glGetUniformLocation(computeProg_, "uPanelOffset");
    uOffsetSignLoc_ = glGetUniformLocation(computeProg_, "uOffsetSign");
    uSweepDirLoc_ = glGetUniformLocation(computeProg_, "uSweepDirection");
    uSwapSinCosLoc_ = glGetUniformLocation(computeProg_, "uSwapSinCos");
    uPhaseOffsetLoc_ = glGetUniformLocation(computeProg_, "uPhaseOffset");

    const int found = (uPanelOffsetLoc_ >= 0) + (uOffsetSignLoc_ >= 0) +
                      (uSweepDirLoc_ >= 0) + (uSwapSinCosLoc_ >= 0) +
                      (uPhaseOffsetLoc_ >= 0);
    if (found > 0 && found < 5) {
        fprintf(stderr,
                "Slicer: partial calibration uniform set detected "
                "(panel=%d, sign=%d, sweep=%d, swap=%d, phase=%d)\n",
                uPanelOffsetLoc_, uOffsetSignLoc_, uSweepDirLoc_,
                uSwapSinCosLoc_, uPhaseOffsetLoc_);
    }
}

void Slicer::setCalibrationParams(float panelOffset,
                                  float offsetSign,
                                  float sweepDirection,
                                  bool swapSinCos,
                                  float phaseOffset)
{
    panelOffset_ = panelOffset;
    offsetSign_ = offsetSign;
    sweepDirection_ = sweepDirection;
    swapSinCos_ = swapSinCos;
    phaseOffset_ = phaseOffset;
}

bool Slicer::init(const char* shaderPath)
{
    if (!loadComputeShader(shaderPath)) return false;
    cacheUniformLocations();

    // Create 2D-array output image texture (128 x 64 x SLICE_COUNT, RGBA8)
    glGenTextures(1, &sliceOutTex_);
    glBindTexture(GL_TEXTURE_2D_ARRAY, sliceOutTex_);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, SLICE_W, SLICE_H, SLICE_COUNT,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    // Create pixel pack buffer for a single whole-array CPU readback
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_PACK_BUFFER, SLICE_W * SLICE_H * SLICE_COUNT * 4,
                 nullptr, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "Slicer: GL error after init: 0x%x\n", err);
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------
// Phase 1: dispatch compute for all angles and initiate DMA to PBO.
// Returns immediately without waiting for GPU completion.
// -----------------------------------------------------------------------
void Slicer::kickDispatch(GLuint voxelTexID)
{
    glUseProgram(computeProg_);

    if (uPanelOffsetLoc_ >= 0) glUniform1f(uPanelOffsetLoc_, panelOffset_);
    if (uOffsetSignLoc_ >= 0) glUniform1f(uOffsetSignLoc_, offsetSign_);
    if (uSweepDirLoc_ >= 0) glUniform1f(uSweepDirLoc_, sweepDirection_);
    if (uSwapSinCosLoc_ >= 0) glUniform1i(uSwapSinCosLoc_, swapSinCos_ ? 1 : 0);
    if (uPhaseOffsetLoc_ >= 0) glUniform1f(uPhaseOffsetLoc_, phaseOffset_);

    // Bind 3D voxel texture to sampler unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, voxelTexID);

    // Bind 2D-array output image to image unit 1 (layered write, RGBA8)
    glBindImageTexture(1, sliceOutTex_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);

    // Single dispatch covering all 128×64×SLICE_COUNT invocations
    glDispatchCompute(2, 16, SLICE_COUNT);

    // Barrier: ensure image writes are visible for the upcoming texture read
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_UPDATE_BARRIER_BIT);

    // Kick off async DMA from GPU texture into PBO — does not block
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    glGetTextureImage(sliceOutTex_, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                      SLICE_W * SLICE_H * SLICE_COUNT * 4, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glUseProgram(0);
}

// -----------------------------------------------------------------------
// Phase 2: wait for GPU/DMA to complete and copy results into buf.
// Must be called after a prior kickDispatch().
// -----------------------------------------------------------------------
void Slicer::syncReadback(SliceBuffer& buf)
{
    glFinish();  // wait for compute + DMA to complete

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    void* ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(buf.data, ptr, SLICE_W * SLICE_H * SLICE_COUNT * 4);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void Slicer::shutdown()
{
    if (computeProg_) { glDeleteProgram(computeProg_);     computeProg_ = 0; }
    if (sliceOutTex_) { glDeleteTextures(1, &sliceOutTex_); sliceOutTex_ = 0; }
    if (pbo_)         { glDeleteBuffers(1, &pbo_);          pbo_         = 0; }
}
