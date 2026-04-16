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

class ReturnToMenuWatcher {
public:
    // ~500ms at ~40fps.
    static constexpr int HOLD_FRAMES = 20;
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
