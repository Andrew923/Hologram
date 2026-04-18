#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include "../engine/CameraConfig.h"
#include <cstdint>

// -----------------------------------------------------------------------
// ParticleApp — floating particles interacting with a red index-finger
// cursor via generic inverse-square forces and local repulsion.
//
// Depth recovery:
//   - If config/camera.json is present and valid, the cursor's Z is
//     computed via pinhole unprojection using the user's wrist→index-MCP
//     reference length.
//   - Otherwise falls back to a hand-size proxy (same bone-pixel metric
//     normalized against an empirical baseline).
//
// Gestures:
//   POINT     → attract particles toward the cursor.
//   FIST      → repel particles away from the cursor.
//   PINCH     → spawn a particle at the cursor (rate-limited).
//   THUMBS_UP → return to menu (via ReturnToMenuWatcher).
// -----------------------------------------------------------------------
class ParticleApp : public IApplication {
public:
    ParticleApp() = default;
    ~ParticleApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer&)                    override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

    static constexpr int N_PARTICLES = 8;

private:
    struct Particle {
        float x, y, z;
        float vx, vy, vz;
        float size;
        uint8_t r, g, b;
    };

    Particle particles_[N_PARTICLES];

    // Smoothed cursor position in voxel space.
    float curX_ = 64.0f;
    float curY_ = 32.0f;
    float curZ_ = 64.0f;
    bool  cursorValid_ = false;

    // One-euro-ish smoothing for Z (bone-length is noisy).
    float smoothedZ_ = 64.0f;

    CameraConfig cam_;
    bool camOk_ = false;

    int spawnCooldown_ = 0;

    ReturnToMenuWatcher menuWatcher_;

    void resetParticles();
    void computeCursor(const SharedHandData& hand);
};
