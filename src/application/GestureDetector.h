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
// any rotational orientation in the image plane. Tests that rely on a
// fixed image axis don't work, and pure tip-distance ratios fail for the
// thumb (curled thumbs rotate across the palm rather than fold back, so
// CMC→tip distance stays similar to CMC→MCP × 3).
//
// Instead we test the angle at each joint via the dot product of the two
// adjacent bone segments:
//
//   straight finger  → bones are collinear, cos(angle) ≈ +1
//   90° bent         → cos = 0
//   fully curled     → bones fold back, cos < 0
//
// A finger counts as extended only when BOTH joints (PIP and DIP for
// fingers; MCP and IP for the thumb) are above EXT_COS_THRESHOLD. This
// rejects partially-bent fingers and the false-thumb case.
//
// PINCH (thumb-tip ↔ index-tip 2D distance) is already
// rotation-invariant; threshold tightened to filter transients.
// -----------------------------------------------------------------------

inline constexpr float PINCH_DIST_THRESHOLD = 0.04f;  // normalised image units
inline constexpr float EXT_COS_THRESHOLD    = 0.6f;   // cos > 0.6  ⇒  joint < 53°

// Cosine of the angle between bone (a→b) and bone (b→c). +1 = collinear
// (straight), 0 = right angle, −1 = folded back. Returns 1.0 for
// degenerate (zero-length) segments so they're treated as straight.
inline float gd_boneCos(const SharedHandData& h, int a, int b, int c)
{
    float ax = h.lm_x[b] - h.lm_x[a];
    float ay = h.lm_y[b] - h.lm_y[a];
    float cx = h.lm_x[c] - h.lm_x[b];
    float cy = h.lm_y[c] - h.lm_y[b];
    float la = sqrtf(ax * ax + ay * ay);
    float lc = sqrtf(cx * cx + cy * cy);
    if (la < 1e-3f || lc < 1e-3f) return 1.0f;
    return (ax * cx + ay * cy) / (la * lc);
}

// Non-thumb finger extended ⇔ both PIP and DIP joints nearly straight.
// Pass the four landmark indices for that finger (e.g. index = 5/6/7/8).
inline bool gd_fingerExtended(const SharedHandData& h,
                              int mcp, int pip, int dip, int tip)
{
    return gd_boneCos(h, mcp, pip, dip) > EXT_COS_THRESHOLD &&
           gd_boneCos(h, pip, dip, tip) > EXT_COS_THRESHOLD;
}

// Thumb (CMC=1, MCP=2, IP=3, Tip=4): both MCP and IP joints nearly
// straight. Distance ratios fail here because curling the thumb across
// the palm doesn't shorten CMC→tip much — the tip just rotates rather
// than folding back over the base.
inline bool gd_thumbExtended(const SharedHandData& h)
{
    return gd_boneCos(h, 1, 2, 3) > EXT_COS_THRESHOLD &&
           gd_boneCos(h, 2, 3, 4) > EXT_COS_THRESHOLD;
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
    bool indexExt  = gd_fingerExtended(hand, 5,  6,  7,  8);
    bool middleExt = gd_fingerExtended(hand, 9,  10, 11, 12);
    bool ringExt   = gd_fingerExtended(hand, 13, 14, 15, 16);
    bool pinkyExt  = gd_fingerExtended(hand, 17, 18, 19, 20);

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
