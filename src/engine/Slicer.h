#pragma once
#include <epoxy/gl.h>
#include <cstdint>
#include <cstring>

static constexpr int SLICE_COUNT   = 240;
static constexpr int SLICE_W       = 128;
static constexpr int SLICE_H       = 64;
static constexpr int PANEL_OFFSET  = 12;  // dead-zone: 48mm gap / 2mm P2 pitch / 2

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

    // Optional calibration controls used by shaders that declare matching uniforms.
    // Legacy shaders that do not declare these uniforms are unaffected.
    void setCalibrationParams(float panelOffset,
                              float offsetSign,
                              float sweepDirection,
                              bool swapSinCos,
                              float phaseOffset);

    // Phase 2: wait for GPU/DMA to complete and copy results into buf.
    // Must be called after a prior kickDispatch().
    void syncReadback(SliceBuffer& buf);

    void shutdown();

private:
    bool loadComputeShader(const char* path);
    void cacheUniformLocations();

    GLuint computeProg_  = 0;  // Compute shader program
    GLuint sliceOutTex_  = 0;  // 2D-array RGBA8 128x64xSLICE_COUNT output image
    GLuint pbo_          = 0;  // Pixel pack buffer for async readback

    // Optional uniform locations for calibration shader variants
    GLint uPanelOffsetLoc_   = -1;
    GLint uOffsetSignLoc_    = -1;
    GLint uSweepDirLoc_      = -1;
    GLint uSwapSinCosLoc_    = -1;
    GLint uPhaseOffsetLoc_   = -1;

    // Runtime values; chosen to match current default slicer behavior.
    float panelOffset_       = PANEL_OFFSET;
    float offsetSign_        = 1.0f;
    float sweepDirection_    = 1.0f;
    bool  swapSinCos_        = false;
    float phaseOffset_       = 0.0f;
};
