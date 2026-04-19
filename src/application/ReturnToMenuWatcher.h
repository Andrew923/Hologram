#pragma once
// -----------------------------------------------------------------------
// ReturnToMenuWatcher — small helper that flags "return to menu" after
// a sustained THUMBS_UP gesture.
//
// Usage inside an IApplication:
//   ReturnToMenuWatcher menuWatcher_;
//   ...
//   void update(const SharedHandData& hand) override {
//       menuWatcher_.update(hand);
//       ...
//   }
//   const char* requestedApp() const override {
//       return menuWatcher_.shouldReturn() ? "menu" : nullptr;
//   }
//
// The watcher only triggers after THUMBS_UP has been held for
// HOLD_FRAMES consecutive frames, which prevents a single brief gesture
// from bouncing between menu and app. After triggering it auto-resets
// so the next menu entry press isn't instantly cancelled by the same
// held gesture.
// -----------------------------------------------------------------------
#include "GestureDetector.h"
#include "VoxelPaint.h"

class ReturnToMenuWatcher {
public:
    // Match menu launch hold time so entering/leaving app feels symmetric.
    static constexpr int HOLD_FRAMES = 60;
    // After triggering, ignore gestures until the user goes through
    // any non-THUMBS_UP state (prevents accidental double-trigger).
    static constexpr int COOLDOWN_FRAMES = 60;

    void update(const SharedHandData& hand) {
        if (triggered_) {
            // Already flagged for this "press" — wait for release.
            if (detectGesture(hand) != Gesture::THUMBS_UP) {
                cooldown_--;
                if (cooldown_ <= 0) {
                    triggered_ = false;
                    heldFrames_ = 0;
                }
            } else {
                cooldown_ = COOLDOWN_FRAMES;
            }
            return;
        }

        if (detectGesture(hand) == Gesture::THUMBS_UP) {
            heldFrames_++;
            if (heldFrames_ >= HOLD_FRAMES) {
                triggered_ = true;
                cooldown_ = COOLDOWN_FRAMES;
            }
        } else {
            heldFrames_ = 0;
        }
    }

    bool shouldReturn() const { return triggered_; }

    float progress01() const {
        float p = (float)heldFrames_ / (float)HOLD_FRAMES;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        return p;
    }

    bool isLoading() const { return heldFrames_ > 0 && !triggered_; }

    void drawLoadingIndicator(uint8_t* voxels) const {
        if (!isLoading()) return;

        constexpr int SIZE = 9;
        constexpr int FILL_AREA_SIZE = SIZE - 2;  // interior size excluding border
        const int ox = VOXEL_W - SIZE - 3;
        const int oy = 2;
        const int oz = VOXEL_D - SIZE - 3;

        for (int i = 0; i < SIZE; ++i) {
            voxpaint::paintVoxel(voxels, ox + i, oy, oz, 40, 120, 200);
            voxpaint::paintVoxel(voxels, ox + i, oy, oz + SIZE - 1, 40, 120, 200);
            voxpaint::paintVoxel(voxels, ox, oy, oz + i, 40, 120, 200);
            voxpaint::paintVoxel(voxels, ox + SIZE - 1, oy, oz + i, 40, 120, 200);
        }

        int fill = (int)(progress01() * FILL_AREA_SIZE + 0.5f);
        for (int z = 0; z < fill; ++z) {
            for (int x = 0; x < FILL_AREA_SIZE; ++x) {
                voxpaint::paintVoxel(voxels, ox + 1 + x, oy, oz + 1 + z, 0, 180, 255);
            }
        }

        int perimeter = 4 * (SIZE - 1);
        int step = (heldFrames_ / 2) % perimeter;
        int sx = 0, sz = 0;
        if (step < (SIZE - 1)) {
            sx = step; sz = 0;
        } else if (step < 2 * (SIZE - 1)) {
            sx = SIZE - 1; sz = step - (SIZE - 1);
        } else if (step < 3 * (SIZE - 1)) {
            sx = (SIZE - 1) - (step - 2 * (SIZE - 1)); sz = SIZE - 1;
        } else {
            sx = 0; sz = (SIZE - 1) - (step - 3 * (SIZE - 1));
        }
        voxpaint::paintVoxel(voxels, ox + sx, oy, oz + sz, 255, 255, 255);
    }

    // Called by the app right before yielding; resets the flag so a
    // subsequent re-enter doesn't immediately re-trigger.
    void acknowledge() {
        triggered_ = false;
        heldFrames_ = 0;
        cooldown_ = COOLDOWN_FRAMES;
    }

private:
    int  heldFrames_ = 0;
    int  cooldown_   = 0;
    bool triggered_  = false;
};
