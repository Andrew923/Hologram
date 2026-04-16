#pragma once
#include <epoxy/gl.h>
#include <cstdint>
#include <cstring>

static constexpr int SLICE_COUNT = 240;
static constexpr int SLICE_W     = 128;
static constexpr int SLICE_H     = 64;

// Buffer holding all SLICE_COUNT slices of 128x64 RGBA8 data
struct SliceBuffer {
    uint8_t data[SLICE_COUNT][SLICE_H][SLICE_W][4];
};

class Slicer {
public:
    Slicer() = default;
    ~Slicer();

    // Load and compile slice_compute.glsl; create slice output texture + PBO
    bool init(const char* shaderPath);

    // Phase 1: dispatch compute shader + initiate DMA to PBO; returns immediately.
    // Call syncReadback() on a subsequent frame to retrieve the data.
    void kickDispatch(GLuint voxelTexID);

    // Phase 2: wait for GPU/DMA to complete and copy results into buf.
    // Must be called after a prior kickDispatch().
    void syncReadback(SliceBuffer& buf);

    void shutdown();

private:
    bool loadComputeShader(const char* path);

    GLuint computeProg_  = 0;  // Compute shader program
    GLuint sliceOutTex_  = 0;  // 2D-array RGBA8 128x64xSLICE_COUNT output image
    GLuint pbo_          = 0;  // Pixel pack buffer for async readback
};
