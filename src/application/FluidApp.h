#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>
#include <chrono>

// 3D FLIP/PIC particle fluid. Replaces the old 2D height-map sim.
// Renders by splatting live particles into the same RGBA8 voxel volume
// the Slicer already consumes; downstream pipeline is unchanged.
class FluidApp : public IApplication {
public:
    FluidApp() = default;
    ~FluidApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer& renderer)           override;
    void teardown(Renderer&)                override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

    // ----- Tunables (public so free helpers in FluidApp.cpp can read them) -
    static constexpr int   PARTICLE_COUNT = 60000;
    static constexpr int   GRID_W = 64;
    static constexpr int   GRID_H = 32;
    static constexpr int   GRID_D = 64;
    static constexpr int   PRESSURE_ITERATIONS = 40;
    static constexpr float FLIP_ALPHA   = 0.97f;
    static constexpr float TILT_K       = 1.5f;
    static constexpr float GRAVITY_MAG  = 28.0f;  // voxel/sec^2 (~9.8 m/s² scaled to voxels)
    static constexpr float DT_MIN       = 1.0f / 120.0f;
    static constexpr float DT_MAX       = 1.0f / 30.0f;

private:

    // ----- Resource lifecycle ----------------------------------------------
    bool   loadProgram(const char* path, GLuint& outProg);
    void   createGridTextures();
    void   initParticleBuffer();

    // ----- Per-frame dispatch helpers --------------------------------------
    void   bindGridImages_Clear();
    void   bindGridImages_P2G();
    void   bindGridImages_Normalize();
    void   bindGridImages_MarkCells();
    void   bindGridImages_Divergence();
    void   bindGridImages_Jacobi(GLuint pIn, GLuint pOut);
    void   bindGridImages_SubtractGrad(GLuint pSource);
    void   bindGridImages_G2P();
    void   bindParticleSSBO();
    void   barrier();
    void   dispatchParticles();
    void   dispatchGrid();

    ReturnToMenuWatcher menuWatcher_;

    // ----- Programs --------------------------------------------------------
    GLuint progClear_       = 0;
    GLuint progRecycle_     = 0;
    GLuint progP2G_         = 0;
    GLuint progNormalize_   = 0;
    GLuint progMark_        = 0;
    GLuint progDivergence_  = 0;
    GLuint progJacobi_      = 0;
    GLuint progSubGrad_     = 0;
    GLuint progG2P_         = 0;
    GLuint progSplat_       = 0;

    // ----- Buffers / textures ---------------------------------------------
    GLuint particleSSBO_ = 0;

    GLuint texAVelX_ = 0, texAVelY_ = 0, texAVelZ_ = 0, texAWeight_ = 0;
    GLuint texVelX_  = 0, texVelY_  = 0, texVelZ_  = 0;
    GLuint texVelXSave_ = 0, texVelYSave_ = 0, texVelZSave_ = 0;
    GLuint texWeightF_ = 0;
    GLuint texPressureA_ = 0, texPressureB_ = 0;
    GLuint texDivergence_ = 0;
    GLuint texCellType_   = 0;

    // ----- Cached uniform locations ---------------------------------------
    GLint uRecycleCount_ = -1, uRecycleFrame_ = -1, uRecycleFill_ = -1;
    GLint uP2GCount_ = -1;
    GLint uNormGravity_ = -1, uNormDt_ = -1;
    GLint uG2PCount_ = -1, uG2PDt_ = -1, uG2PAlpha_ = -1;
    GLint uSplatCount_ = -1, uSplatDt_ = -1;

    // ----- CPU-side state -------------------------------------------------
    float fingerSmoothX_ = 0.5f;
    float fingerSmoothY_ = 0.5f;
    bool  fingerActive_  = false;
    uint32_t frameCounter_ = 0;
    std::chrono::steady_clock::time_point lastTick_{};
    bool   firstFrame_ = true;
};
