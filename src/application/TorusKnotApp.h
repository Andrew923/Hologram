#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"

class TorusKnotApp : public IApplication {
public:
    TorusKnotApp() = default;
    ~TorusKnotApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    float phase_      = 0.0f;
    float angularVel_ = 0.0f;
    float scale_      = 1.0f;

    ReturnToMenuWatcher menuWatcher_;
};

