#include "MenuApp.h"
#include "VoxelPaint.h"
#include "GestureDetector.h"
#include "../engine/Renderer.h"
#include "../engine/JsonConfig.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr int   THUMBS_UP_LAUNCH_FRAMES = 60;   // ~1.5s at 40fps
static constexpr float SMOOTHING     = 0.35f;
static constexpr float SPEED_SCALE   = 0.80f;  // angular velocity per unit finger offset
static constexpr float ANGULAR_DECAY = 0.90f;
static constexpr float CAROUSEL_RADIUS = 0.70f;   // model-space radius
static constexpr float ICON_SIZE_BASE  = 0.18f;   // model-space icon scale
static constexpr float ICON_SIZE_HIGHLIGHT = 0.28f;

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Normalize an angle to the range [-pi, pi].
static inline float wrapAngle(float a) {
    while (a >  (float)M_PI) a -= 2.0f * (float)M_PI;
    while (a < -(float)M_PI) a += 2.0f * (float)M_PI;
    return a;
}

int MenuApp::iconKindForId(const std::string& id) const
{
    if (id == "cube")      return 0;
    if (id == "torus")     return 1;
    if (id == "particles") return 2;
    if (id == "wireframe") return 3;
    if (id == "corridor")  return 4;
    if (id == "city")      return 5;
    if (id == "morph")     return 6;
    return 0;
}

void MenuApp::loadEntries(const std::string& path)
{
    entries_.clear();

    JsonValue root;
    bool loaded = loadJsonFile(path, root);

    std::string scanDir;
    if (loaded) {
        const auto& arr = root.at("entries").asArray();
        for (const auto& e : arr) {
            Entry ent;
            ent.id    = e.at("id").asString();
            ent.label = e.at("label").asString(ent.id);
            ent.obj   = e.at("obj").asString();
            ent.iconKind = iconKindForId(ent.id);
            if (ent.id.empty()) continue;
            entries_.push_back(std::move(ent));
        }
        scanDir = root.at("scan_models_dir").asString();
    } else {
        fprintf(stderr, "MenuApp: no menu.json, using defaults\n");
    }

    // Fallback defaults if no entries were loaded.
    if (entries_.empty()) {
        entries_.push_back({"cube",      "Cube",      "",  0});
        entries_.push_back({"torus",     "Torus",     "",  1});
        entries_.push_back({"particles", "Particles", "",  2});
    }

    // Optionally scan a directory for .obj / .glb files and append wireframe entries
    // for any not already explicitly listed.
    if (!scanDir.empty()) {
        std::error_code ec;
        if (std::filesystem::is_directory(scanDir, ec)) {
            for (const auto& de : std::filesystem::directory_iterator(scanDir, ec)) {
                if (!de.is_regular_file()) continue;
                auto p = de.path();
                auto ext = p.extension().string();
                // Lowercase the extension for case-insensitive comparison.
                for (auto& ch : ext)
                    ch = static_cast<char>(tolower(static_cast<unsigned char>(ch)));
                if (ext != ".obj" && ext != ".glb") continue;
                std::string rel = p.string();
                bool already = false;
                for (const auto& e : entries_) {
                    if (e.id == "wireframe" && e.obj == rel) { already = true; break; }
                }
                if (already) continue;
                Entry ent;
                ent.id = "wireframe";
                ent.label = p.stem().string();
                ent.obj = rel;
                ent.iconKind = 3;
                entries_.push_back(std::move(ent));
            }
        }
    }

    fprintf(stderr, "MenuApp: %zu carousel entries loaded\n", entries_.size());
}

void MenuApp::setup(Renderer& /*renderer*/)
{
    loadEntries("config/menu.json");
    carouselAngle_ = 0.0f;
    angularVel_    = 0.0f;
    thumbsUpHeld_  = 0;
    pendingLaunch_.clear();
}

int MenuApp::selectedIndex() const
{
    if (entries_.empty()) return 0;
    int N = (int)entries_.size();
    // We place item i at baseAngle = 2pi * i / N + carouselAngle_. The
    // selected item is the one closest to angle 0 (front center).
    float best = 1e9f;
    int   idx  = 0;
    for (int i = 0; i < N; ++i) {
        float a = wrapAngle(2.0f * (float)M_PI * i / N + carouselAngle_);
        float d = fabsf(a);
        if (d < best) { best = d; idx = i; }
    }
    return idx;
}

void MenuApp::update(const SharedHandData& hand)
{
    if (!pendingLaunch_.empty()) return;  // launch pending; stop processing
    if (entries_.empty()) return;

    if (!hand.hand_detected) {
        angularVel_ *= ANGULAR_DECAY;
        carouselAngle_ += angularVel_;
        thumbsUpHeld_ = 0;
        return;
    }

    // Scroll: index finger tip X position drives angular velocity.
    // Left of center (x < 0.5) = clockwise; right of center = counter-clockwise.
    float offset = hand.lm_x[8] - 0.5f;
    float target = -offset * SPEED_SCALE;
    angularVel_ += (target - angularVel_) * SMOOTHING;
    carouselAngle_ += angularVel_;

    // THUMBS_UP held → launch selected.
    Gesture g = detectGesture(hand);
    if (g == Gesture::THUMBS_UP) {
        thumbsUpHeld_++;
        if (thumbsUpHeld_ >= THUMBS_UP_LAUNCH_FRAMES) {
            const Entry& sel = entries_[selectedIndex()];
            if (sel.id == "wireframe" && !sel.obj.empty()) {
                pendingLaunch_ = "wireframe:" + sel.obj;
            } else {
                pendingLaunch_ = sel.id;
            }
            fprintf(stderr, "MenuApp: launching '%s'\n", pendingLaunch_.c_str());
        }
    } else {
        thumbsUpHeld_ = 0;
    }
}

// Draw a tiny icon (model-space half-extent s) centered at (cx,cy,cz).
// kind: 0=cube wireframe, 1=torus knot, 2=dots cloud, 3=tetrahedron.
void MenuApp::drawIcon(uint8_t* voxels, int kind, float cx, float cy, float cz,
                       float s, bool highlighted) const
{
    // Color tables (indexed by kind)
    static const uint8_t icon_colors[7][3] = {
        {  0, 255, 255},  // 0 cube: cyan
        {255,  60, 200},  // 1 torus knot: magenta-ish
        {255, 180,  40},  // 2 dots: amber
        { 80, 255, 120},  // 3 tetra: mint
        {  0, 220, 180},  // 4 corridor: teal
        {255, 160,  30},  // 5 city: orange-gold
        {255, 180,   0},  // 6 morph: amber-gold
    };
    static const uint8_t highlight[3] = {255, 255, 255};
    int colorIdx = (kind < 7) ? kind : 0;
    const uint8_t* rgb = highlighted ? highlight : icon_colors[colorIdx];

    auto mapToVoxel = [&](float mx, float my, float mz,
                          int& vx, int& vy, int& vz) {
        vx = (int)roundf((cx + mx * s) * 0.5f * (VOXEL_W - 1) + (VOXEL_W - 1) * 0.5f);
        vy = (int)roundf((cy + my * s) * 0.5f * (VOXEL_H - 1) + (VOXEL_H - 1) * 0.5f);
        vz = (int)roundf((cz + mz * s) * 0.5f * (VOXEL_D - 1) + (VOXEL_D - 1) * 0.5f);
    };

    if (kind == 0) {
        // Cube wireframe (tiny)
        static const float v[8][3] = {
            {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
            {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
        };
        static const int e[12][2] = {
            {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        int px[8], py[8], pz[8];
        for (int i = 0; i < 8; ++i) mapToVoxel(v[i][0], v[i][1], v[i][2], px[i], py[i], pz[i]);
        for (int i = 0; i < 12; ++i)
            voxpaint::paint3DLine(voxels, px[e[i][0]], py[e[i][0]], pz[e[i][0]],
                                         px[e[i][1]], py[e[i][1]], pz[e[i][1]],
                                         rgb[0], rgb[1], rgb[2]);
    } else if (kind == 1) {
        // Mini torus knot
        const int N = 36;
        const int P = 2, Q = 3;
        const float R = 0.68f, r = 0.28f;
        int prevX = 0, prevY = 0, prevZ = 0;
        bool havePrev = false;
        for (int i = 0; i <= N; ++i) {
            float t = (float)i / (float)N * 2.0f * (float)M_PI;
            float ring = R + r * cosf((float)Q * t);
            float x = ring * cosf((float)P * t);
            float y = 0.9f * r * sinf((float)Q * t);
            float z = ring * sinf((float)P * t);
            int vx, vy, vz;
            mapToVoxel(x, y, z, vx, vy, vz);
            if (havePrev) {
                voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz,
                                      rgb[0], rgb[1], rgb[2]);
            }
            prevX = vx; prevY = vy; prevZ = vz;
            havePrev = true;
        }
    } else if (kind == 2) {
        // Cluster of dots
        static const float pts[8][3] = {
            {0.0f, 0.0f, 0.0f},
            {0.6f, 0.2f, -0.3f},
            {-0.5f, 0.5f, 0.4f},
            {0.3f, -0.6f, 0.2f},
            {-0.3f, -0.3f, -0.6f},
            {0.7f, -0.1f, 0.6f},
            {-0.7f, 0.1f, -0.2f},
            {0.1f, 0.7f, -0.5f},
        };
        for (int i = 0; i < 8; ++i) {
            int vx, vy, vz;
            mapToVoxel(pts[i][0], pts[i][1], pts[i][2], vx, vy, vz);
            voxpaint::paintCube(voxels, vx, vy, vz, 0, rgb[0], rgb[1], rgb[2]);
        }
    } else if (kind == 3) {
        // Tetrahedron
        static const float v[4][3] = {
            { 0.0f,  1.0f,  0.0f},
            {-0.9f, -0.5f, -0.5f},
            { 0.9f, -0.5f, -0.5f},
            { 0.0f, -0.5f,  0.9f},
        };
        int px[4], py[4], pz[4];
        for (int i = 0; i < 4; ++i) mapToVoxel(v[i][0], v[i][1], v[i][2], px[i], py[i], pz[i]);
        static const int e[6][2] = {{0,1},{0,2},{0,3},{1,2},{2,3},{3,1}};
        for (int i = 0; i < 6; ++i)
            voxpaint::paint3DLine(voxels, px[e[i][0]], py[e[i][0]], pz[e[i][0]],
                                         px[e[i][1]], py[e[i][1]], pz[e[i][1]],
                                         rgb[0], rgb[1], rgb[2]);
    } else if (kind == 4) {
        // Corridor: open-top U-shape with a slight curve suggesting a winding hallway
        // Two vertical wall pillars connected by a floor line; bent path on the floor
        static const float wallL[2][3] = {{-0.7f, -1.0f, -0.5f}, {-0.7f,  0.7f, -0.5f}};
        static const float wallR[2][3] = {{ 0.7f, -1.0f, -0.5f}, { 0.7f,  0.7f, -0.5f}};
        static const float floorPts[3][3] = {
            {-0.7f, -1.0f, -0.5f},
            {  0.0f, -1.0f,  0.0f},  // midpoint offset to suggest curve
            { 0.7f, -1.0f,  0.5f},
        };
        int ax, ay, az, bx, by_, bz;
        mapToVoxel(wallL[0][0], wallL[0][1], wallL[0][2], ax, ay, az);
        mapToVoxel(wallL[1][0], wallL[1][1], wallL[1][2], bx, by_, bz);
        voxpaint::paint3DLine(voxels, ax, ay, az, bx, by_, bz, rgb[0], rgb[1], rgb[2]);
        mapToVoxel(wallR[0][0], wallR[0][1], wallR[0][2], ax, ay, az);
        mapToVoxel(wallR[1][0], wallR[1][1], wallR[1][2], bx, by_, bz);
        voxpaint::paint3DLine(voxels, ax, ay, az, bx, by_, bz, rgb[0], rgb[1], rgb[2]);
        // Floor segments (two-part bent line)
        int fpx[3], fpy[3], fpz[3];
        for (int i = 0; i < 3; ++i) mapToVoxel(floorPts[i][0], floorPts[i][1], floorPts[i][2], fpx[i], fpy[i], fpz[i]);
        voxpaint::paint3DLine(voxels, fpx[0], fpy[0], fpz[0], fpx[1], fpy[1], fpz[1], rgb[0], rgb[1], rgb[2]);
        voxpaint::paint3DLine(voxels, fpx[1], fpy[1], fpz[1], fpx[2], fpy[2], fpz[2], rgb[0], rgb[1], rgb[2]);
    } else if (kind == 6) {
        // Morph: octahedron icon (double-pyramid, visually distinct)
        static const float v[6][3] = {
            { 1.0f,  0.0f,  0.0f}, {-1.0f,  0.0f,  0.0f},
            { 0.0f,  1.0f,  0.0f}, { 0.0f, -1.0f,  0.0f},
            { 0.0f,  0.0f,  1.0f}, { 0.0f,  0.0f, -1.0f},
        };
        static const int e[12][2] = {
            {0,2},{0,3},{0,4},{0,5},
            {1,2},{1,3},{1,4},{1,5},
            {2,4},{2,5},{3,4},{3,5},
        };
        int px[6], py[6], pz[6];
        for (int i = 0; i < 6; ++i) mapToVoxel(v[i][0], v[i][1], v[i][2], px[i], py[i], pz[i]);
        for (int i = 0; i < 12; ++i)
            voxpaint::paint3DLine(voxels, px[e[i][0]], py[e[i][0]], pz[e[i][0]],
                                         px[e[i][1]], py[e[i][1]], pz[e[i][1]],
                                         rgb[0], rgb[1], rgb[2]);
    } else if (kind == 5) {
        // City: three buildings of different heights side by side
        static const float buildings[3][4] = {
            // {x_center, z_center, half_w, height}  (model space)
            {-0.65f,  0.0f, 0.20f,  0.5f},   // left: medium height
            { 0.00f,  0.0f, 0.22f,  1.0f},   // center: tall
            { 0.65f,  0.0f, 0.18f,  0.3f},   // right: short
        };
        for (int b = 0; b < 3; ++b) {
            float bcx = buildings[b][0], bcz = buildings[b][1];
            float bhw = buildings[b][2], bh  = buildings[b][3];
            float x0m = bcx - bhw, x1m = bcx + bhw;
            float z0m = bcz - 0.18f, z1m = bcz + 0.18f;
            float yBot = -1.0f, yTop = -1.0f + bh * 2.0f;
            int px0, py0, pz0, px1, py1, pz1;
            mapToVoxel(x0m, yBot, z0m, px0, py0, pz0);
            mapToVoxel(x1m, yTop, z1m, px1, py1, pz1);
            // Four vertical pillars
            voxpaint::paint3DLine(voxels, px0, py0, pz0, px0, py1, pz0, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px1, py0, pz0, px1, py1, pz0, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px1, py0, pz1, px1, py1, pz1, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px0, py0, pz1, px0, py1, pz1, rgb[0], rgb[1], rgb[2]);
            // Top edges
            voxpaint::paint3DLine(voxels, px0, py1, pz0, px1, py1, pz0, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px1, py1, pz0, px1, py1, pz1, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px1, py1, pz1, px0, py1, pz1, rgb[0], rgb[1], rgb[2]);
            voxpaint::paint3DLine(voxels, px0, py1, pz1, px0, py1, pz0, rgb[0], rgb[1], rgb[2]);
        }
    }
}

void MenuApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    if (entries_.empty()) { renderer.uploadVoxelBuffer(voxels); return; }

    int N = (int)entries_.size();
    int sel = selectedIndex();

    // Draw all icons first.
    for (int i = 0; i < N; ++i) {
        float a = 2.0f * (float)M_PI * i / N + carouselAngle_;
        float cx = CAROUSEL_RADIUS * sinf(a);
        float cz = CAROUSEL_RADIUS * cosf(a);
        // Lift selected item slightly so it's unambiguous.
        float cy = (i == sel) ? 0.08f : 0.0f;

        float s = (i == sel) ? ICON_SIZE_HIGHLIGHT : ICON_SIZE_BASE;

        drawIcon(voxels, entries_[i].iconKind, cx, cy, cz, s, i == sel);
    }

    // Rising blue fill on selected icon: scan the selected icon's voxel
    // bounding box and recolor any set voxels that fall in the bottom
    // (progress * VOXEL_H) rows to blue. y=0 is top in this display,
    // so "bottom" = high y indices; threshold sweeps downward as progress grows.
    if (thumbsUpHeld_ > 0) {
        float progress = thumbsUpHeld_ / (float)THUMBS_UP_LAUNCH_FRAMES;
        if (progress > 1.0f) progress = 1.0f;
        // threshold: voxels with y < threshold are in the "bottom" region
        // (y=0 is now bottom of display after slicer Y-flip)
        int threshold = (int)(progress * VOXEL_H);

        float a   = 2.0f * (float)M_PI * sel / N + carouselAngle_;
        float icx = CAROUSEL_RADIUS * sinf(a);
        float icy = 0.08f;
        float icz = CAROUSEL_RADIUS * cosf(a);
        float s   = ICON_SIZE_HIGHLIGHT;

        // Bounding box of the selected icon in voxel space.
        int vxc = (int)roundf(icx * 0.5f * (VOXEL_W - 1) + (VOXEL_W - 1) * 0.5f);
        int vyc = (int)roundf(icy * 0.5f * (VOXEL_H - 1) + (VOXEL_H - 1) * 0.5f);
        int vzc = (int)roundf(icz * 0.5f * (VOXEL_D - 1) + (VOXEL_D - 1) * 0.5f);
        int hvx = (int)ceilf(s * 0.5f * (VOXEL_W - 1));
        int hvy = (int)ceilf(s * 0.5f * (VOXEL_H - 1));
        int hvz = (int)ceilf(s * 0.5f * (VOXEL_D - 1));

        int x0 = std::max(0,         vxc - hvx), x1 = std::min(VOXEL_W - 1, vxc + hvx);
        int y0 = std::max(0,         vyc - hvy), y1 = std::min(threshold - 1, vyc + hvy);
        int z0 = std::max(0,         vzc - hvz), z1 = std::min(VOXEL_D - 1, vzc + hvz);

        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    int idx = ((z * VOXEL_H + y) * VOXEL_W + x) * 4;
                    if (voxels[idx + 3] != 0) {
                        voxels[idx + 0] = 0;
                        voxels[idx + 1] = 80;
                        voxels[idx + 2] = 255;
                    }
                }
            }
        }
    }

    renderer.uploadVoxelBuffer(voxels);
}
