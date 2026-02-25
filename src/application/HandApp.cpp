#include "HandApp.h"
#include "../engine/Renderer.h"
#include <cstring>
#include <cstdlib>
#include <ctime>

// constexpr definitions (required in some C++17 contexts)
constexpr int HandApp::CONNECTIONS[20][2];
constexpr int HandApp::FINGERTIP_INDICES[5];

static double nowSeconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool isFingertip(int idx) {
    for (int t : {4, 8, 12, 16, 20})
        if (idx == t) return true;
    return false;
}

// -----------------------------------------------------------------------
// Bresenham line drawing on 128×64 BGR canvas
// -----------------------------------------------------------------------
void HandApp::drawPixel(uint8_t* canvas, int x, int y,
                         uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= 128 || y < 0 || y >= 64) return;
    int idx = (y * 128 + x) * 3;
    canvas[idx + 0] = b;   // BGR storage
    canvas[idx + 1] = g;
    canvas[idx + 2] = r;
}

void HandApp::drawLine(uint8_t* canvas,
                        int x0, int y0, int x1, int y1,
                        uint8_t r, uint8_t g, uint8_t b)
{
    int dx =  abs(x1 - x0);
    int dy =  abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        drawPixel(canvas, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void HandApp::setup(Renderer& /*renderer*/)
{
    // Initialize all OneEuroFilter instances with plan parameters
    for (int i = 0; i < 21; ++i) {
        filtersX_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
        filtersY_[i] = OneEuroFilter(1.0f, 0.5f, 1.0f);
    }
}

void HandApp::update(const SharedHandData& hand)
{
    anyValid_ = false;
    if (!hand.hand_detected) return;

    // Check wrist is non-zero
    if (hand.lm_x[0] == 0.0f && hand.lm_y[0] == 0.0f) return;

    anyValid_ = true;
    double t = hand.timestamp > 0.0 ? hand.timestamp : nowSeconds();

    for (int i = 0; i < 21; ++i) {
        // Scale normalized [0,1] landmark to canvas pixels
        float rawX = hand.lm_x[i] * 128.0f;
        float rawY = hand.lm_y[i] * 64.0f;

        smoothX_[i] = filtersX_[i].filter(rawX, t);
        smoothY_[i] = filtersY_[i].filter(rawY, t);
    }
}

void HandApp::draw(Renderer& /*renderer*/)
{
    // Clear canvas to black
    memset(canvas_, 0, sizeof(canvas_));

    if (anyValid_) {
        // Draw skeleton connections (white bones)
        for (auto& conn : CONNECTIONS) {
            int j1 = conn[0], j2 = conn[1];
            int px1 = (int)smoothX_[j1], py1 = (int)smoothY_[j1];
            int px2 = (int)smoothX_[j2], py2 = (int)smoothY_[j2];
            // Skip if both endpoints are at origin (undetected)
            if ((px1 == 0 && py1 == 0) || (px2 == 0 && py2 == 0)) continue;
            drawLine(canvas_, px1, py1, px2, py2, 255, 255, 255);
        }

        // Draw joints
        for (int i = 0; i < 21; ++i) {
            int px = (int)smoothX_[i];
            int py = (int)smoothY_[i];
            if (px == 0 && py == 0) continue;

            if (isFingertip(i)) {
                // Red for fingertips
                drawPixel(canvas_, px, py, 255, 0, 0);
            } else {
                // Green for other joints
                drawPixel(canvas_, px, py, 0, 255, 0);
            }
        }
    }

    // Send via network (HandApp owns UDP send; bypasses slicer)
    network_.sendHandFrame(frameID_++, canvas_);
}
