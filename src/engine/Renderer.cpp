#include "Renderer.h"
#include <cstdio>
#include <cstring>

// -----------------------------------------------------------------------
// EGL extension function pointers (loaded after eglGetDisplay succeeds)
// -----------------------------------------------------------------------
static PFNEGLQUERYDEVICESEXTPROC      s_eglQueryDevicesEXT      = nullptr;
static PFNEGLGETPLATFORMDISPLAYEXTPROC s_eglGetPlatformDisplayEXT = nullptr;

Renderer::~Renderer() { shutdown(); }

// -----------------------------------------------------------------------
// EGL headless initialization
// -----------------------------------------------------------------------
bool Renderer::initEGL()
{
    // Try to load EGL_EXT_platform_device extension
    s_eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    s_eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (s_eglQueryDevicesEXT && s_eglGetPlatformDisplayEXT) {
        EGLint numDevices = 0;
        s_eglQueryDevicesEXT(0, nullptr, &numDevices);
        if (numDevices > 0) {
            EGLDeviceEXT device = nullptr;
            s_eglQueryDevicesEXT(1, &device, &numDevices);
            eglDisplay_ = s_eglGetPlatformDisplayEXT(
                EGL_PLATFORM_DEVICE_EXT, device, nullptr);
        }
    }

    // Fallback: default display (works on L4T without $DISPLAY)
    if (eglDisplay_ == EGL_NO_DISPLAY) {
        eglDisplay_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }

    if (eglDisplay_ == EGL_NO_DISPLAY) {
        fprintf(stderr, "Renderer: no EGL display available\n");
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(eglDisplay_, &major, &minor)) {
        fprintf(stderr, "Renderer: eglInitialize failed\n");
        return false;
    }
    fprintf(stderr, "Renderer: EGL %d.%d\n", major, minor);

    // Bind desktop OpenGL API (required for compute shaders)
    if (!eglBindAPI(EGL_OPENGL_API)) {
        fprintf(stderr, "Renderer: eglBindAPI(EGL_OPENGL_API) failed\n");
        return false;
    }

    // Choose config for pbuffer (headless — no window surface needed)
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    EGLConfig config;
    EGLint    numConfigs = 0;
    if (!eglChooseConfig(eglDisplay_, configAttribs, &config, 1, &numConfigs)
        || numConfigs < 1) {
        fprintf(stderr, "Renderer: eglChooseConfig failed\n");
        return false;
    }

    // Create a tiny 1×1 pbuffer surface (headless placeholder)
    static const EGLint pbufAttribs[] = {
        EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE
    };
    eglSurface_ = eglCreatePbufferSurface(eglDisplay_, config, pbufAttribs);
    if (eglSurface_ == EGL_NO_SURFACE) {
        fprintf(stderr, "Renderer: eglCreatePbufferSurface failed\n");
        return false;
    }

    // Try OpenGL 4.5 Core Profile, fall back to 4.3 minimum
    const int versions[][2] = {{4,5},{4,4},{4,3}};
    for (auto& v : versions) {
        const EGLint ctxAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, v[0],
            EGL_CONTEXT_MINOR_VERSION, v[1],
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE
        };
        eglContext_ = eglCreateContext(eglDisplay_, config, EGL_NO_CONTEXT, ctxAttribs);
        if (eglContext_ != EGL_NO_CONTEXT) {
            fprintf(stderr, "Renderer: OpenGL %d.%d Core context created\n", v[0], v[1]);
            break;
        }
    }
    if (eglContext_ == EGL_NO_CONTEXT) {
        fprintf(stderr, "Renderer: eglCreateContext failed (need at least GL 4.3)\n");
        return false;
    }

    if (!eglMakeCurrent(eglDisplay_, eglSurface_, eglSurface_, eglContext_)) {
        fprintf(stderr, "Renderer: eglMakeCurrent failed\n");
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------
// OpenGL resource initialization
// -----------------------------------------------------------------------
bool Renderer::initGL()
{
    // GLEW must be initialized after making the context current
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "Renderer: glewInit failed: %s\n", glewGetErrorString(glewErr));
        return false;
    }
    // Drain any spurious GL error from glewInit
    while (glGetError() != GL_NO_ERROR) {}

    fprintf(stderr, "Renderer: GL vendor: %s\n", glGetString(GL_VENDOR));
    fprintf(stderr, "Renderer: GL renderer: %s\n", glGetString(GL_RENDERER));
    fprintf(stderr, "Renderer: GL version: %s\n", glGetString(GL_VERSION));

    // ----------------------------------------------------------------
    // Create 3D voxel texture: 128(X) x 64(Y) x 128(Z), RGBA8
    // Texture coordinates: (s,t,r) -> (X/128, Y/64, Z/128) normalized [0,1]
    // ----------------------------------------------------------------
    glGenTextures(1, &voxelTex_);
    glBindTexture(GL_TEXTURE_3D, voxelTex_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    static const float borderColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // Allocate storage (no initial data — will be cleared/uploaded before use)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8,
                 VOXEL_W, VOXEL_H, VOXEL_D,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_3D, 0);

    // ----------------------------------------------------------------
    // Create layered FBO (all Z-layers attached for geometry pass)
    // ----------------------------------------------------------------
    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    // Attach the full 3D texture as a layered color attachment
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, voxelTex_, 0);
    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Renderer: layered FBO incomplete: 0x%x\n", status);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "Renderer: GL error after initGL: 0x%x\n", err);
        return false;
    }

    return true;
}

bool Renderer::init()
{
    if (!initEGL()) return false;
    if (!initGL())  return false;
    return true;
}

void Renderer::clearVoxels()
{
    // glClearTexImage requires GL 4.4+
    static const uint8_t zero[4] = {0, 0, 0, 0};
    glClearTexImage(voxelTex_, 0, GL_RGBA, GL_UNSIGNED_BYTE, zero);
}

void Renderer::uploadVoxelBuffer(const uint8_t* data)
{
    // data layout: data[(z*VOXEL_H + y)*VOXEL_W + x] -> pixel (x,y,z)
    glBindTexture(GL_TEXTURE_3D, voxelTex_);
    glTexSubImage3D(GL_TEXTURE_3D, 0,
                    0, 0, 0,
                    VOXEL_W, VOXEL_H, VOXEL_D,
                    GL_RGBA, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_3D, 0);
}

void Renderer::bindFBO()   { glBindFramebuffer(GL_FRAMEBUFFER, fbo_); }
void Renderer::unbindFBO() { glBindFramebuffer(GL_FRAMEBUFFER, 0);    }

void Renderer::shutdown()
{
    if (voxelTex_) { glDeleteTextures(1, &voxelTex_); voxelTex_ = 0; }
    if (fbo_)      { glDeleteFramebuffers(1, &fbo_);  fbo_      = 0; }

    if (eglDisplay_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (eglContext_ != EGL_NO_CONTEXT) {
            eglDestroyContext(eglDisplay_, eglContext_);
            eglContext_ = EGL_NO_CONTEXT;
        }
        if (eglSurface_ != EGL_NO_SURFACE) {
            eglDestroySurface(eglDisplay_, eglSurface_);
            eglSurface_ = EGL_NO_SURFACE;
        }
        eglTerminate(eglDisplay_);
        eglDisplay_ = EGL_NO_DISPLAY;
    }
}
