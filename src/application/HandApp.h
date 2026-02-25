#pragma once
#include "IApplication.h"
#include "../engine/Network.h"
#include <cstdint>
#include <cmath>
#include <ctime>

// -----------------------------------------------------------------------
// One Euro Filter (matches 2d_send.py implementation)
// -----------------------------------------------------------------------
class OneEuroFilter {
public:
    OneEuroFilter(float minCutoff = 1.0f, float beta = 0.5f, float dCutoff = 1.0f)
        : minCutoff_(minCutoff), beta_(beta), dCutoff_(dCutoff)
        , xPrev_(0.0f), dxPrev_(0.0f), tPrev_(0.0), initialized_(false) {}

    float filter(float x, double t) {
        if (!initialized_) {
            initialized_ = true;
            xPrev_ = x;
            tPrev_ = t;
            return x;
        }
        float dt = (float)(t - tPrev_);
        if (dt <= 0.0f) dt = 1e-6f;

        float dx    = (x - xPrev_) / dt;
        float ad    = smoothFactor(dCutoff_, dt);
        float dxHat = ad * dx + (1.0f - ad) * dxPrev_;

        float cutoff = minCutoff_ + beta_ * fabsf(dxHat);
        float a      = smoothFactor(cutoff, dt);
        float xHat   = a * x + (1.0f - a) * xPrev_;

        xPrev_  = xHat;
        dxPrev_ = dxHat;
        tPrev_  = t;
        return xHat;
    }

    void reset() { initialized_ = false; }

private:
    float smoothFactor(float cutoff, float dt) const {
        float r = 2.0f * (float)M_PI * cutoff * dt;
        return r / (r + 1.0f);
    }

    float  minCutoff_, beta_, dCutoff_;
    float  xPrev_, dxPrev_;
    double tPrev_;
    bool   initialized_;
};

// -----------------------------------------------------------------------
// HandApp: translates 2d_send.py
// -----------------------------------------------------------------------
class HandApp : public IApplication {
public:
    explicit HandApp(Network& network) : network_(network) {}
    ~HandApp() override = default;

    void setup(Renderer&)                    override;
    void update(const SharedHandData& hand)  override;
    void draw(Renderer&)                     override;
    void teardown(Renderer&)                 override {}
    bool bypassSlicer() const                override { return true; }

private:
    Network& network_;

    // One Euro Filters: 21 joints × 2 axes = 42 instances
    OneEuroFilter filtersX_[21];
    OneEuroFilter filtersY_[21];

    // Smoothed joint positions in canvas space [0,128) × [0,64)
    float smoothX_[21] = {};
    float smoothY_[21] = {};
    bool  anyValid_ = false;

    // 128×64 BGR canvas
    uint8_t canvas_[64 * 128 * 3] = {};

    // Frame counter (wraps 0-255)
    uint8_t frameID_ = 0;

    // Hand skeleton connections (MediaPipe native order)
    static constexpr int CONNECTIONS[20][2] = {
        {0,1},{1,2},{2,3},{3,4},           // thumb
        {0,5},{5,6},{6,7},{7,8},           // index
        {0,9},{9,10},{10,11},{11,12},      // middle
        {0,13},{13,14},{14,15},{15,16},    // ring
        {0,17},{17,18},{18,19},{19,20}     // pinky
    };
    static constexpr int FINGERTIP_INDICES[5] = {4, 8, 12, 16, 20};

    static void drawLine(uint8_t* canvas,
                         int x0, int y0, int x1, int y1,
                         uint8_t r, uint8_t g, uint8_t b);
    static void drawPixel(uint8_t* canvas, int x, int y,
                          uint8_t r, uint8_t g, uint8_t b);
};
