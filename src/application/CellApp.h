#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>

// 3D cellular automaton on a 64×32×64 grid. Each alive cell paints a 2×2×2
// voxel block (matches the slicer's 2-voxel = 1-grid scale used elsewhere
// in the project). The CA only steps every UPDATE_INTERVAL frames so motion
// reads as deliberate pulsing rather than noise; the fingertip pinches a
// new seed cluster into existence at the user's hand position.
//
// Default rule is 3D-Life B5,6,7/S5,6,7 (Carter Bays family) — symmetric,
// produces oscillating clumps and slow-moving structures.
class CellApp : public IApplication {
public:
    CellApp() = default;
    ~CellApp() override = default;

    static constexpr int GRID_W = 64, GRID_H = 32, GRID_D = 64;
    static constexpr int UPDATE_INTERVAL = 6;       // ~5 Hz at 30fps Jetson
    static constexpr int SEED_INTERVAL   = 4;       // pinch re-seeds every N frames
    static constexpr float SEED_RADIUS   = 4.0f;    // grid cells

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer& renderer)           override;
    void teardown(Renderer&)                override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    bool loadProgram(const char* path, GLuint& outProg);
    void seedSphere(int gx, int gy, int gz, float radius, bool clear);

    ReturnToMenuWatcher menuWatcher_;

    GLuint progStep_   = 0;
    GLuint progRender_ = 0;
    GLuint progSeed_   = 0;

    GLuint texCur_  = 0;
    GLuint texNext_ = 0;

    GLint uStepBirth_   = -1, uStepSurvive_ = -1;
    GLint uSeedCenter_  = -1, uSeedRadius_  = -1, uSeedSeed_ = -1, uSeedClear_ = -1;
    GLint uRenderColor_ = -1;

    uint32_t frameCounter_ = 0;
    uint32_t seedCounter_  = 0;

    // B5,6,7 / S5,6,7 (Carter Bays' rule family for 3D Life)
    uint32_t birthMask_   = (1u << 5) | (1u << 6) | (1u << 7);
    uint32_t surviveMask_ = (1u << 5) | (1u << 6) | (1u << 7);

    bool  fingerActive_  = false;
    bool  pinchActive_   = false;
    int   fingerGridX_   = GRID_W / 2;
    int   fingerGridY_   = GRID_H / 2;
    int   fingerGridZ_   = GRID_D / 2;
};
