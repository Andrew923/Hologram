#pragma once
#include "IApplication.h"

class FluidApp : public IApplication {
public:
    FluidApp() = default;
    ~FluidApp() override = default;

    void setup(Renderer&)                    override {}
    void update(const SharedHandData&)       override {}
    void draw(Renderer&)                     override {}
    void teardown(Renderer&)                 override {}
    bool bypassSlicer() const                override { return false; }
};
