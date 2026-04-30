#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>

// Voxel paint: index fingertip leaves a glowing 3D trail. Trail decays over
// a couple seconds and color-cycles with age (white → cyan → magenta → blue
// → off). Forgiving with system latency: the trail is naturally a
// long-exposure of your hand, so the lag reads as intentional.
//
// Cursor mapping:
//   X = lm[8].x * 128                       (image-X → bowl-X)
//   Z = lm[8].y * 128                       (image-Y → bowl-Z)
//   Y = thumb-index pinch openness          (tight = low, wide = high)
class PaintApp : public IApplication {
public:
    PaintApp() = default;
    ~PaintApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer& renderer)           override;
    void teardown(Renderer&)                override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    bool loadProgram(const char* path, GLuint& outProg);

    ReturnToMenuWatcher menuWatcher_;
    GLuint progStep_ = 0;
    GLuint texAge_   = 0;

    GLint uCursorLoc_       = -1;
    GLint uCursorRadiusLoc_ = -1;
    GLint uCursorActiveLoc_ = -1;
    GLint uDecayLoc_        = -1;

    bool  fingerActive_ = false;
    float cursorX_      = 64.0f;
    float cursorY_      = 32.0f;
    float cursorZ_      = 64.0f;
};
