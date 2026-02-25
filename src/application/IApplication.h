#pragma once
#include "../../shared_defs.h"

class Renderer;  // forward declare

class IApplication {
public:
    virtual ~IApplication() = default;

    virtual void setup(Renderer&)        = 0;
    virtual void update(const SharedHandData&) = 0;
    virtual void draw(Renderer&)         = 0;    // HandApp sends UDP here
    virtual void teardown(Renderer&)     {}

    // Return true if this app sends UDP directly (skip voxel/slicer pipeline)
    virtual bool bypassSlicer() const    { return false; }
};
