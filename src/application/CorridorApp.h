#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"

class CorridorApp : public IApplication {
public:
    CorridorApp() = default;
    ~CorridorApp() override = default;

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
    float camX_ = 0.0f;    // smoothed world X (left-right)
    float camZ_ = 64.0f;   // smoothed world Z (depth)
};
