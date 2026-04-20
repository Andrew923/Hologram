#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"

class CityApp : public IApplication {
public:
    CityApp() = default;
    ~CityApp() override = default;

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
    float camX_ = 100.0f;  // smoothed world X
    float camZ_ = 100.0f;  // smoothed world Z
};
