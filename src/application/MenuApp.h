#pragma once
#include "IApplication.h"
#include <cstdint>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// MenuApp — 3D carousel of application icons.
//
// Membership is loaded from config/menu.json at setup() time; if the
// file is absent, a default list (cube, torus, particles) is used.
//
// Interaction:
//   PINCH (rising edge) → smoothly scroll one slot forward.
//   FIVE_FINGERS (open palm, fingers splayed, held ~1.5s) → launch the
//     centered item; a blue floor rises from the bottom while held as a
//     visual progress indicator. Launch target is exposed via
//     requestedApp(); main.cpp performs the swap.
//
// For wireframe entries, requestedApp() returns "wireframe:<obj-path>"
// so main.cpp can call WireframeApp::setModel() before setup().
// -----------------------------------------------------------------------
class MenuApp : public IApplication {
public:
    MenuApp() = default;
    ~MenuApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return pendingLaunch_.empty() ? nullptr : pendingLaunch_.c_str();
    }

private:
    struct Entry {
        std::string id;        // "cube", "torus", "particles", "wireframe"
        std::string label;     // human-readable (currently unused — iconography only)
        std::string obj;       // only for "wireframe" entries
        int iconKind = 0;      // 0=cube, 1=torus knot, 2=dots, 3=tetra
    };

    std::vector<Entry> entries_;

    // Carousel rotation state (radians). Each frame carouselAngle_ eases
    // toward targetAngle_; pinch shifts the target by -2π/N.
    float carouselAngle_ = 0.0f;
    float targetAngle_   = 0.0f;

    // How long the launch gesture has been held (frames).
    int launchHeld_ = 0;

    // Pinch edge detection — pinch (rising edge) snaps the carousel one
    // slot forward. Cooldown prevents a single sustained pinch or jittery
    // detection from firing rapidly.
    bool pinchPrev_     = false;
    int  pinchCooldown_ = 0;

    // Set by update() when the launch is triggered. main.cpp is expected
    // to consume this via requestedApp() and then the app is torn down;
    // no explicit clearing is needed (MenuApp is re-setup on re-entry).
    std::string pendingLaunch_;

    void loadEntries(const std::string& path);
    int  iconKindForId(const std::string& id) const;
    int  selectedIndex() const;
    void drawIcon(uint8_t* voxels, int kind, float cx, float cy, float cz,
                  float s, bool highlighted) const;
};
