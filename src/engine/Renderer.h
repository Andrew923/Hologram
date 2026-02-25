#pragma once
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glew.h>
#include <cstdint>

// Voxel grid dimensions: 128(X) x 64(Y) x 128(Z)
static constexpr int VOXEL_W = 128;
static constexpr int VOXEL_H = 64;
static constexpr int VOXEL_D = 128;
static constexpr int VOXEL_BYTES = VOXEL_W * VOXEL_H * VOXEL_D * 4; // RGBA8 = 4MB

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    // Initialize EGL headless context + 3D voxel texture + layered FBO
    bool init();

    // Clear all voxels to (0,0,0,0) — uses glClearTexImage (GL 4.4+)
    void clearVoxels();

    // Upload a CPU-side RGBA8 voxel buffer (VOXEL_BYTES bytes)
    // Layout: data[(z*VOXEL_H + y)*VOXEL_W + x] = voxel at (x,y,z)
    void uploadVoxelBuffer(const uint8_t* data);

    GLuint getVoxelTextureID() const { return voxelTex_; }

    void bindFBO();
    void unbindFBO();

    void shutdown();

private:
    bool initEGL();
    bool initGL();

    EGLDisplay  eglDisplay_  = EGL_NO_DISPLAY;
    EGLContext  eglContext_  = EGL_NO_CONTEXT;
    EGLSurface  eglSurface_  = EGL_NO_SURFACE;

    GLuint voxelTex_ = 0;   // GL_TEXTURE_3D, RGBA8, 128x64x128
    GLuint fbo_      = 0;   // Layered framebuffer (for GL_LINES geometry pass)
};
