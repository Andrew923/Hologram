#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>
#include <chrono>

// Strange-attractor particle flow. ~8000 particles all integrate the same
// chaotic ODE in lockstep; the resulting cloud traces the attractor's
// shape. Pinch swaps to the next attractor and re-seeds. No real-time
// twitch interaction needed — the spectacle is the trajectory itself.
class FlowApp : public IApplication {
public:
    FlowApp() = default;
    ~FlowApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer& renderer)           override;
    void teardown(Renderer&)                override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

    static constexpr int PARTICLE_COUNT = 8000;
    static constexpr int NUM_ATTRACTORS = 4;       // Lorenz, Aizawa, Halvorsen, Thomas

private:
    bool loadProgram(const char* path, GLuint& outProg);
    void initParticles(int attractorIdx);
    void getTransform(int idx, float origin[3], float scale[3],
                      float& dt, int& substeps, float color[4]) const;

    ReturnToMenuWatcher menuWatcher_;
    GLuint progStep_     = 0;
    GLuint progSplat_    = 0;
    GLuint particleSSBO_ = 0;

    GLint uStepCount_      = -1;
    GLint uStepDt_         = -1;
    GLint uStepSubsteps_   = -1;
    GLint uStepAttractor_  = -1;
    GLint uSplatCount_     = -1;
    GLint uSplatOrigin_    = -1;
    GLint uSplatScale_     = -1;
    GLint uSplatColor_     = -1;

    int   attractorIdx_     = 0;
    bool  pinchPrev_        = false;
    int   pinchHoldFrames_  = 0;
    bool  fingerActive_     = false;
    float fingerSmoothX_    = 0.5f;
    float fingerSmoothY_    = 0.5f;
};
