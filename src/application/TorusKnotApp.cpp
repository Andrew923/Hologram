#include "TorusKnotApp.h"
#include "VoxelPaint.h"
#include "GestureDetector.h"
#include "../engine/Renderer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr int   N_STEPS = 320;

static constexpr float TORUS_MAJOR     = 0.62f;
static constexpr float TUBE_RADIUS_PX  = 3.0f;

// Auto-spin: ~1 rev per 7 s at 30 fps.
static constexpr float AUTO_SPIN_VEL   = 0.03f;

// How fast the rendered parameters glide toward the finger-driven targets.
static constexpr float PARAM_SMOOTHING = 0.15f;

// Per-finger parameter ranges.
static constexpr int   P_MIN = 1, P_MAX = 5;
static constexpr int   Q_MIN = 1, Q_MAX = 5;
static constexpr float TUBE_MIN = 0.06f, TUBE_MAX = 0.36f;
static constexpr float HEIGHT_MIN = 0.20f, HEIGHT_MAX = 1.60f;

// Voxel-space scale (constant — the bowl already frames the knot).
static constexpr float SCALE_PX = 18.0f;
static constexpr float SCALE    = SCALE_PX / (VOXEL_H - 1);

static const uint8_t KNOT_A_RGB[3] = {255,  60, 220};
static const uint8_t KNOT_B_RGB[3] = {  0, 220, 255};

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// Continuous "how extended" value for a non-thumb finger.
// Averages the cosines at PIP and DIP, remaps from [-1,1] to [0,1] and
// clamps. 0 ≈ fully curled (folded back), 1 ≈ perfectly straight.
static float fingerExtCont(const SharedHandData& h,
                           int mcp, int pip, int dip, int tip)
{
    float c1 = gd_boneCos(h, mcp, pip, dip);
    float c2 = gd_boneCos(h, pip, dip, tip);
    float t  = (c1 + c2) * 0.5f;
    return clampf((t + 1.0f) * 0.5f, 0.0f, 1.0f);
}

void TorusKnotApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
}

void TorusKnotApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    // Auto-spin runs regardless of whether the hand is present.
    phase_ += AUTO_SPIN_VEL;

    if (!hand.hand_detected) return;

    float tIndex  = fingerExtCont(hand, 5,  6,  7,  8);
    float tMiddle = fingerExtCont(hand, 9,  10, 11, 12);
    float tRing   = fingerExtCont(hand, 13, 14, 15, 16);
    float tPinky  = fingerExtCont(hand, 17, 18, 19, 20);

    float tgtP    = (float)P_MIN + tIndex  * (float)(P_MAX - P_MIN);
    float tgtQ    = (float)Q_MIN + tMiddle * (float)(Q_MAX - Q_MIN);
    float tgtTube = TUBE_MIN     + tRing   * (TUBE_MAX   - TUBE_MIN);
    float tgtH    = HEIGHT_MIN   + tPinky  * (HEIGHT_MAX - HEIGHT_MIN);

    pCont_      = lerp(pCont_,      tgtP,    PARAM_SMOOTHING);
    qCont_      = lerp(qCont_,      tgtQ,    PARAM_SMOOTHING);
    tubeRadius_ = lerp(tubeRadius_, tgtTube, PARAM_SMOOTHING);
    heightGain_ = lerp(heightGain_, tgtH,    PARAM_SMOOTHING);
}

void TorusKnotApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    std::memset(voxels, 0, sizeof(voxels));

    // P/Q must be integers for the knot to close; snap from the smoothed
    // continuous values.
    const int P = std::max(P_MIN, std::min(P_MAX, (int)std::round(pCont_)));
    const int Q = std::max(Q_MIN, std::min(Q_MAX, (int)std::round(qCont_)));
    const float r = tubeRadius_;
    const float h = heightGain_;

    int prevX = 0, prevY = 0, prevZ = 0;
    bool havePrev = false;

    const float cp = std::cos(phase_);
    const float sp = std::sin(phase_);

    for (int i = 0; i <= N_STEPS; ++i) {
        float u = (float)i / (float)N_STEPS;
        float t = u * 2.0f * (float)M_PI;

        float cqt = std::cos((float)Q * t);
        float sqt = std::sin((float)Q * t);
        float cpt = std::cos((float)P * t);
        float spt = std::sin((float)P * t);

        float ring = TORUS_MAJOR + r * cqt;
        float x = ring * cpt;
        float y = h * r * sqt;
        float z = ring * spt;

        // Auto-spin around the vertical (Y) axis.
        float xr =  cp * x - sp * z;
        float zr =  sp * x + cp * z;

        int vx = (int)std::round((xr * SCALE + 1.0f) * 0.5f * (VOXEL_W - 1));
        int vy = (int)std::round((y  * SCALE + 1.0f) * 0.5f * (VOXEL_H - 1));
        int vz = (int)std::round((zr * SCALE + 1.0f) * 0.5f * (VOXEL_D - 1));
        vx = std::max(0, std::min(VOXEL_W - 1, vx));
        vy = std::max(0, std::min(VOXEL_H - 1, vy));
        vz = std::max(0, std::min(VOXEL_D - 1, vz));

        const uint8_t* col = (i & 1) ? KNOT_A_RGB : KNOT_B_RGB;
        if (havePrev) {
            voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz,
                                  col[0], col[1], col[2]);
        }
        voxpaint::paintSphere(voxels, vx, vy, vz, TUBE_RADIUS_PX,
                              col[0], col[1], col[2]);

        prevX = vx; prevY = vy; prevZ = vz;
        havePrev = true;
    }

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
