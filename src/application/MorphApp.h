#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <cstdint>

class MorphApp : public IApplication {
public:
    MorphApp() = default;
    ~MorphApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    void teardown(Renderer&)                override {}
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    ReturnToMenuWatcher menuWatcher_;

    float rotX_     = 0.0f;
    float rotY_     = 0.0f;
    float spinVelX_ = 0.0f;
    float spinVelY_ = 0.02f;
    float morphT_   = 0.0f;   // [0,1]: 0=tetrahedron, 1=dodecahedron
    float posX_     = 0.0f;   // offset from grid center in voxel units
    float posZ_     = 0.0f;
};
