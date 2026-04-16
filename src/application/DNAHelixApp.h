#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <cstdint>

// -----------------------------------------------------------------------
// DNAHelixApp — renders a rotating DNA-style double helix.
//
// Interaction:
//   PEACE (index+middle extended, horizontal direction) → rotate about Y.
//     Pointing right spins one way, left the other.
//   PINCH (thumb↔index distance)                        → scale the helix.
//   THUMBS_UP (held ~0.5s)                               → return to menu.
//   No hand/gesture                                      → angular velocity
//                                                          decays.
//
// Geometry is rotationally symmetric about Y so the rotation is always
// visually coherent.
// -----------------------------------------------------------------------
class DNAHelixApp : public IApplication {
public:
    DNAHelixApp() = default;
    ~DNAHelixApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

private:
    // Smoothed rotation about the vertical (Y) axis.
    float phase_      = 0.0f;   // accumulated Y-rotation (radians)
    float angularVel_ = 0.0f;   // radians per frame, smoothed

    // Scale — same mapping as CubeApp's pinch-to-scale.
    float scale_ = 1.0f;

    ReturnToMenuWatcher menuWatcher_;
};
