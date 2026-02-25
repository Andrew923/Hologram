#pragma once
#include "IApplication.h"

class PongApp : public IApplication {
public:
    PongApp() = default;
    ~PongApp() override = default;

    void setup(Renderer&)                    override {}
    void update(const SharedHandData&)       override {}
    void draw(Renderer&)                     override {}
    void teardown(Renderer&)                 override {}
    bool bypassSlicer() const                override { return false; }
};
