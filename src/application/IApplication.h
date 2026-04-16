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

    // If non-null, main.cpp will teardown this app and switch to the named
    // one after the current frame. Return values:
    //   "cube", "dna", "particles", "menu", "hand", "pong", "fluid"
    //   "wireframe:<path-to-obj>"  (main.cpp calls setModel() then setup())
    // Implementations should return nullptr once the request is accepted
    // (i.e. after one frame of observation by the harness).
    virtual const char* requestedApp() const { return nullptr; }
};
