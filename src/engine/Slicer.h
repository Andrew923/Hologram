#pragma once
#include <GL/glew.h>
#include <cstdint>
#include <cstring>

static constexpr int SLICE_COUNT = 120;
static constexpr int SLICE_W     = 128;
static constexpr int SLICE_H     = 64;

// Buffer holding all 120 slices of 128x64 RGBA8 data
struct SliceBuffer {
    uint8_t data[SLICE_COUNT][SLICE_H][SLICE_W][4];
};

class Slicer {
public:
    Slicer() = default;
    ~Slicer();

    // Load and compile slice_compute.glsl; create slice output texture + PBO
    bool init(const char* shaderPath);

    // Dispatch compute shader for all 120 angles; fill sliceBuffer in place
    void sliceAll(GLuint voxelTexID, SliceBuffer& buf);

    void shutdown();

private:
    bool loadComputeShader(const char* path);

    GLuint computeProg_  = 0;  // Compute shader program
    GLuint sliceOutTex_  = 0;  // 2D RGBA8 128x64 output image
    GLuint pbo_          = 0;  // Pixel pack buffer for async readback

    GLint  locTheta_     = -1; // Uniform location for uTheta
};
