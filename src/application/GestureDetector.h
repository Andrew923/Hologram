#pragma once
#include "../../shared_defs.h"
#include <cmath>

// -----------------------------------------------------------------------
// Gesture enum — all detectable hand gestures.
// -----------------------------------------------------------------------
enum class Gesture {
    NONE,           // no hand / indeterminate
    FIST,           // all fingers curled
    FIST_UP,        // unused (kept for backward compat — orientation isn't
                    //         meaningful with the upward-pointing camera)
    THUMBS_UP,      // only thumb extended
    POINT,          // only index finger extended
    PEACE,          // index + middle extended (V sign)
    ONE_FINGER,     // exactly one non-index/thumb finger extended
    TWO_FINGERS,    // two fingers extended (other than peace pattern)
    THREE_FINGERS,  // three fingers extended
    FOUR_FINGERS,   // four fingers extended
    FIVE_FINGERS,   // all fingers extended (open splayed palm)
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
// Rotation-invariant primitives.
//
// The rig's camera looks straight up, so the user can hold their hand at
// any rotational orientation in the image plane. Tests that rely on
// "fingers point up = smaller lm_y" don't work — instead we compare
// distances along the hand's own structure:
//
//   When a finger is extended, its tip sits roughly the full bone length
//   away from the MCP knuckle (tip ≈ 3× the MCP→PIP distance).
//   When curled, the tip folds back toward the MCP (tip ≈ 1× MCP→PIP).
//
// FINGER_EXT_RATIO is the threshold; landmark distances ratio above it
// counts as extended. Same idea for the thumb (CMC → MCP vs CMC → tip).
// PINCH detection (thumb-tip ↔ index-tip distance) is already
// rotation-invariant so it stays as-is.
// -----------------------------------------------------------------------

inline constexpr float PINCH_DIST_THRESHOLD = 0.05f;  // normalised image units
inline constexpr float FINGER_EXT_RATIO     = 1.7f;   // tip:knuckle distance ratio

inline float gd_lmDist(const SharedHandData& h, int a, int b)
{
    float dx = h.lm_x[a] - h.lm_x[b];
    float dy = h.lm_y[a] - h.lm_y[b];
    return sqrtf(dx * dx + dy * dy);
}

// True if a non-thumb finger is extended. mcp/pip/tip are the MediaPipe
// landmark indices for that finger (e.g. index = 5/6/8).
inline bool gd_fingerExtended(const SharedHandData& h, int mcp, int pip, int tip)
{
    float dPip = gd_lmDist(h, mcp, pip);
    float dTip = gd_lmDist(h, mcp, tip);
    if (dPip < 1e-4f) return false;
    return dTip > FINGER_EXT_RATIO * dPip;
}

// Thumb has a different bone arrangement (CMC=1, MCP=2, IP=3, Tip=4).
// Same logic: extended if the tip is much farther from the CMC base than
// the first knuckle is.
inline bool gd_thumbExtended(const SharedHandData& h)
{
    float dMcp = gd_lmDist(h, 1, 2);
    float dTip = gd_lmDist(h, 1, 4);
    if (dMcp < 1e-4f) return false;
    return dTip > FINGER_EXT_RATIO * dMcp;
}

// -----------------------------------------------------------------------
// detectGesture — classify the current hand pose.
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

    // PINCH: thumb tip(4) close to index tip(8). 2D image distance is
    // already rotation-invariant.
    float pinchDist = hypotf(hand.lm_x[4] - hand.lm_x[8],
                             hand.lm_y[4] - hand.lm_y[8]);
    if (pinchDist < PINCH_DIST_THRESHOLD) return Gesture::PINCH;

    bool thumbExt  = gd_thumbExtended(hand);
    bool indexExt  = gd_fingerExtended(hand, 5,  6,  8);
    bool middleExt = gd_fingerExtended(hand, 9,  10, 12);
    bool ringExt   = gd_fingerExtended(hand, 13, 14, 16);
    bool pinkyExt  = gd_fingerExtended(hand, 17, 18, 20);

    int fingerCount = (int)indexExt + (int)middleExt + (int)ringExt + (int)pinkyExt;

    if (fingerCount == 0 && !thumbExt)        return Gesture::FIST;
    if (fingerCount == 0 && thumbExt)         return Gesture::THUMBS_UP;

    int totalCount = fingerCount + (int)thumbExt;

    if (totalCount == 1 && indexExt)                          return Gesture::POINT;
    if (totalCount == 2 && indexExt && middleExt && !thumbExt) return Gesture::PEACE;

    switch (totalCount) {
        case 1:  return Gesture::ONE_FINGER;
        case 2:  return Gesture::TWO_FINGERS;
        case 3:  return Gesture::THREE_FINGERS;
        case 4:  return Gesture::FOUR_FINGERS;
        case 5:  return Gesture::FIVE_FINGERS;
        default: return Gesture::NONE;
    }
}
