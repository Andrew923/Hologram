#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>

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

private:
    bool loadShader(const char* path);

    ReturnToMenuWatcher menuWatcher_;

    GLuint hTex_[3]   = {0, 0, 0};
    GLuint fluidProg_ = 0;
    int    hCurIdx_   = 0;
    int    hPrevIdx_  = 1;
    int    hNextIdx_  = 2;

    GLint  uFingerXZLoc_     = -1;
    GLint  uImpulseLoc_      = -1;
    GLint  uFingerActiveLoc_ = -1;

    float fingerX_      = 64.0f;
    float fingerZ_      = 64.0f;
    float impulse_      = 0.3f;
    bool  fingerActive_ = false;
};
