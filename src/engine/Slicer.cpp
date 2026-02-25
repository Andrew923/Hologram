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

    locTheta_ = glGetUniformLocation(computeProg_, "uTheta");
    return true;
}

bool Slicer::init(const char* shaderPath)
{
    if (!loadComputeShader(shaderPath)) return false;

    // Create 2D output image texture (128 x 64, RGBA8)
    glGenTextures(1, &sliceOutTex_);
    glBindTexture(GL_TEXTURE_2D, sliceOutTex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SLICE_W, SLICE_H,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create pixel pack buffer for CPU readback
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_PACK_BUFFER, SLICE_W * SLICE_H * 4,
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
// Dispatch compute for 120 angles, readback each slice into buf
// -----------------------------------------------------------------------
void Slicer::sliceAll(GLuint voxelTexID, SliceBuffer& buf)
{
    glUseProgram(computeProg_);

    // Bind 3D voxel texture to sampler unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, voxelTexID);

    // Bind 2D output image to image unit 1 (write-only, RGBA8)
    glBindImageTexture(1, sliceOutTex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    for (int i = 0; i < SLICE_COUNT; ++i) {
        float theta = (float)(i * 2.0 * M_PI / SLICE_COUNT);
        glUniform1f(locTheta_, theta);

        // Dispatch: 2x16x1 workgroups, local_size 64x4 = 128x64 total
        glDispatchCompute(2, 16, 1);

        // Barrier: ensure image writes are visible for texture read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                      | GL_TEXTURE_UPDATE_BARRIER_BIT);

        // Readback via PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_);
        glGetTextureImage(sliceOutTex_, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                          SLICE_W * SLICE_H * 4, 0);
        glFinish();

        void* ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (ptr) {
            memcpy(buf.data[i], ptr, SLICE_W * SLICE_H * 4);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    glUseProgram(0);
}

void Slicer::shutdown()
{
    if (computeProg_) { glDeleteProgram(computeProg_);     computeProg_ = 0; }
    if (sliceOutTex_) { glDeleteTextures(1, &sliceOutTex_); sliceOutTex_ = 0; }
    if (pbo_)         { glDeleteBuffers(1, &pbo_);          pbo_         = 0; }
}
