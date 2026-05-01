#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"

// (P,Q) torus knot, auto-spinning around its vertical axis.
//
// Each non-thumb finger smoothly drives one parameter via its joint
// extension cosine — curl a finger to lower the parameter, straighten
// it to raise it. With 4 fingers you can independently dial 4 knobs:
//
//   index  → P  (loops around the torus axis,  integer 1..5, snaps)
//   middle → Q  (loops through the donut hole, integer 1..5, snaps)
//   ring   → tube radius (continuous, thin → fat)
//   pinky  → vertical squish HEIGHT_GAIN (flat ring → tall barrel)
//
// gcd(P,Q) > 1 produces a torus *link* (multiple separate loops);
// gcd == 1 produces a single knot. (2,3) trefoil, (2,5) cinquefoil,
// (3,4) eight-lobe, etc.
//
// FIVE_FINGERS held → return to menu.
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
    float phase_ = 0.0f;     // accumulated auto-spin angle

    // Smoothed parameter targets driven by per-finger extension.
    float pCont_       = 2.0f;     // index   → P
    float qCont_       = 3.0f;     // middle  → Q
    float tubeRadius_  = 0.22f;    // ring    → minor radius
    float heightGain_  = 0.95f;    // pinky   → Y squish

    ReturnToMenuWatcher menuWatcher_;
};
