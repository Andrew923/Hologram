#pragma once
#include "../../shared_defs.h"
#include <cmath>

// -----------------------------------------------------------------------
// Gesture enum — all detectable hand gestures.
// Designed to be reusable across multiple applications.
// -----------------------------------------------------------------------
enum class Gesture {
    NONE,           // no hand / indeterminate
    FIST,           // all fingers curled, generic orientation
    FIST_UP,        // all fingers curled, wrist below knuckles (fist pointing up)
    THUMBS_UP,      // thumb extended upward, other fingers curled
    POINT,          // only index finger extended
    PEACE,          // index + middle extended (V sign)
    ONE_FINGER,     // 1 non-index/thumb finger extended
    TWO_FINGERS,    // 2 fingers extended (other than peace pattern)
    THREE_FINGERS,  // 3 fingers extended
    FOUR_FINGERS,   // 4 fingers extended
    FIVE_FINGERS,   // all fingers extended (open palm)
    PINCH,          // thumb tip and index tip close together
};

inline const char* gestureName(Gesture g)
{
    switch (g) {
        case Gesture::NONE:          return "NONE";
        case Gesture::FIST:          return "FIST";
        case Gesture::FIST_UP:       return "FIST_UP";
        case Gesture::THUMBS_UP:     return "THUMBS_UP";
        case Gesture::POINT:         return "POINT";
        case Gesture::PEACE:         return "PEACE";
        case Gesture::ONE_FINGER:    return "ONE_FINGER";
        case Gesture::TWO_FINGERS:   return "TWO_FINGERS";
        case Gesture::THREE_FINGERS: return "THREE_FINGERS";
        case Gesture::FOUR_FINGERS:  return "FOUR_FINGERS";
        case Gesture::FIVE_FINGERS:  return "FIVE_FINGERS";
        case Gesture::PINCH:         return "PINCH";
        default:                     return "UNKNOWN";
    }
}

// -----------------------------------------------------------------------
// detectGesture — classify the current hand pose.
//
// Uses raw MediaPipe normalized coordinates (lm_x/lm_y in [0,1]).
//
// IMPORTANT: this rig's camera is mounted under the rotor looking up,
// and after the hand_tracker.py horizontal flip the resulting image has
// world-up at *larger* lm_y values (the inverse of MediaPipe's default
// top-left-origin convention). All the "finger extended" / "wrist below"
// tests below are therefore inverted from how you'd see them written in
// other MediaPipe tutorials. If you re-mount the camera, flip the
// comparison signs back.
//
// Landmark indices (MediaPipe native order):
//   0=wrist
//   1-4=thumb  (CMC, MCP, IP, tip)
//   5-8=index  (MCP, PIP, DIP, tip)
//   9-12=middle(MCP, PIP, DIP, tip)
//  13-16=ring  (MCP, PIP, DIP, tip)
//  17-20=pinky (MCP, PIP, DIP, tip)
// -----------------------------------------------------------------------
inline Gesture detectGesture(const SharedHandData& hand)
{
    if (!hand.hand_detected) return Gesture::NONE;

    // --- Pinch: thumb tip(4) close to index tip(8) ---
    float pinchDist = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
    if (pinchDist < 0.05f) return Gesture::PINCH;

    // --- Finger extension (tip "above" PIP/IP in world = LARGER lm_y
    // because the camera is inverted; see header note). ---
    bool thumbUp  = hand.lm_y[4]  > hand.lm_y[2];
    bool indexUp  = hand.lm_y[8]  > hand.lm_y[6];
    bool middleUp = hand.lm_y[12] > hand.lm_y[10];
    bool ringUp   = hand.lm_y[16] > hand.lm_y[14];
    bool pinkyUp  = hand.lm_y[20] > hand.lm_y[18];

    int fingerCount = (int)indexUp + (int)middleUp + (int)ringUp + (int)pinkyUp;

    // --- Fist variants (no non-thumb fingers extended) ---
    if (fingerCount == 0 && !thumbUp) {
        // Upward fist: wrist(0) is below knuckles in world = wrist.y is
        // SMALLER than middle_mcp.y here.
        bool wristBelow = hand.lm_y[0] < hand.lm_y[9];
        return wristBelow ? Gesture::FIST_UP : Gesture::FIST;
    }

    // --- Thumbs up: only thumb extended ---
    if (thumbUp && fingerCount == 0) return Gesture::THUMBS_UP;

    // --- Specific two-finger patterns (checked before generic count) ---
    int totalCount = fingerCount + (int)thumbUp;

    if (totalCount == 1 && indexUp)                    return Gesture::POINT;
    if (totalCount == 2 && indexUp && middleUp && !thumbUp) return Gesture::PEACE;

    // --- Generic finger count ---
    switch (totalCount) {
        case 1:  return Gesture::ONE_FINGER;
        case 2:  return Gesture::TWO_FINGERS;
        case 3:  return Gesture::THREE_FINGERS;
        case 4:  return Gesture::FOUR_FINGERS;
        case 5:  return Gesture::FIVE_FINGERS;
        default: return Gesture::NONE;
    }
}
