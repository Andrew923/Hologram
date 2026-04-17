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
//   PEACE (index+middle extended, horizontal direction) → spin carousel.
//   THUMBS_UP (held ~0.5s)                               → launch
//     the centered item. The launch target is exposed via
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
        int iconKind = 0;      // 0=cube, 1=helix, 2=dots, 3=tetra
    };

    std::vector<Entry> entries_;

    // Carousel rotation state (radians). Larger carouselAngle_ moves the
    // selected index one way.
    float carouselAngle_  = 0.0f;
    float angularVel_     = 0.0f;

    // How long THUMBS_UP has been held (frames).
    int thumbsUpHeld_ = 0;

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
