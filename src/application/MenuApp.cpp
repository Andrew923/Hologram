#include "MenuApp.h"
#include "VoxelPaint.h"
#include "GestureDetector.h"
#include "../engine/Renderer.h"
#include "../engine/JsonConfig.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr int   THUMBS_UP_LAUNCH_FRAMES = 60;
static constexpr float SMOOTHING       = 0.35f;
static constexpr float SPEED_SCALE     = 0.80f;
static constexpr float ANGULAR_DECAY   = 0.90f;
static constexpr float CAROUSEL_RADIUS = 0.70f;
static constexpr float ICON_SIZE_HIGHLIGHT = 0.28f;
static constexpr int   UNSELECTED_CUBE_HALF = 2;          // voxels
static constexpr int   PINCH_COOLDOWN_FRAMES = 8;          // anti-double-fire

// -----------------------------------------------------------------------
// Icon kinds — keep in sync with iconKindForId() and drawIcon().
// -----------------------------------------------------------------------
namespace {
constexpr int ICON_CUBE      = 0;
constexpr int ICON_TORUS     = 1;
constexpr int ICON_DOTS      = 2;
constexpr int ICON_TETRA     = 3;
constexpr int ICON_CORRIDOR  = 4;
constexpr int ICON_CITY      = 5;
constexpr int ICON_MORPH     = 6;
constexpr int ICON_PYRAMID   = 7;
constexpr int ICON_DROPLET   = 8;
constexpr int ICON_WAVE      = 9;
constexpr int ICON_TEXT      = 10;
}  // namespace

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float wrapAngle(float a) {
    while (a >  (float)M_PI) a -= 2.0f * (float)M_PI;
    while (a < -(float)M_PI) a += 2.0f * (float)M_PI;
    return a;
}

// -----------------------------------------------------------------------
// 3×5 ASCII font for the wireframe text icons. Each glyph is 5 rows × 3
// cols; '#' = lit, '.' = unlit. Indices [0..25] = 'A'..'Z', [26] = '_'.
// -----------------------------------------------------------------------
static const char* const FONT_3x5[27][5] = {
    /* A */ { ".#.", "#.#", "###", "#.#", "#.#" },
    /* B */ { "##.", "#.#", "##.", "#.#", "##." },
    /* C */ { ".##", "#..", "#..", "#..", ".##" },
    /* D */ { "##.", "#.#", "#.#", "#.#", "##." },
    /* E */ { "###", "#..", "##.", "#..", "###" },
    /* F */ { "###", "#..", "##.", "#..", "#.." },
    /* G */ { ".##", "#..", "#.#", "#.#", ".##" },
    /* H */ { "#.#", "#.#", "###", "#.#", "#.#" },
    /* I */ { "###", ".#.", ".#.", ".#.", "###" },
    /* J */ { "..#", "..#", "..#", "#.#", ".#." },
    /* K */ { "#.#", "##.", "#..", "##.", "#.#" },
    /* L */ { "#..", "#..", "#..", "#..", "###" },
    /* M */ { "#.#", "###", "###", "#.#", "#.#" },
    /* N */ { "##.", "#.#", "#.#", "#.#", "#.#" },
    /* O */ { "###", "#.#", "#.#", "#.#", "###" },
    /* P */ { "##.", "#.#", "##.", "#..", "#.." },
    /* Q */ { ".#.", "#.#", "#.#", "###", "..#" },
    /* R */ { "##.", "#.#", "##.", "#.#", "#.#" },
    /* S */ { ".##", "#..", ".#.", "..#", "##." },
    /* T */ { "###", ".#.", ".#.", ".#.", ".#." },
    /* U */ { "#.#", "#.#", "#.#", "#.#", "###" },
    /* V */ { "#.#", "#.#", "#.#", "#.#", ".#." },
    /* W */ { "#.#", "#.#", "###", "###", "#.#" },
    /* X */ { "#.#", "#.#", ".#.", "#.#", "#.#" },
    /* Y */ { "#.#", "#.#", ".#.", ".#.", ".#." },
    /* Z */ { "###", "..#", ".#.", "#..", "###" },
    /* _ */ { "...", "...", "...", "...", "###" },
};

static const char* const* glyphFor(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c >= 'A' && c <= 'Z') return FONT_3x5[c - 'A'];
    if (c == '_')             return FONT_3x5[26];
    return nullptr;
}

// Render up to 4 chars of `text` centred at voxel (xc, yc, zc), one voxel
// thick along Z. Y is screen-up after the slicer flip, so we paint row 0
// at (yc + height/2) and row N-1 at (yc - height/2).
static void drawText(uint8_t* voxels, const std::string& text,
                     int xc, int yc, int zc,
                     uint8_t cr, uint8_t cg, uint8_t cb)
{
    constexpr int CHAR_W = 3, CHAR_H = 5, GAP = 1;
    int n = (int)std::min(text.size(), (size_t)4);
    if (n <= 0) return;
    int totalW = n * CHAR_W + (n - 1) * GAP;
    int xStart = xc - totalW / 2;
    int yTop   = yc + CHAR_H / 2;       // top row in screen-up coordinates

    for (int ci = 0; ci < n; ++ci) {
        const char* const* glyph = glyphFor(text[ci]);
        if (!glyph) continue;
        int charX = xStart + ci * (CHAR_W + GAP);
        for (int row = 0; row < CHAR_H; ++row) {
            for (int col = 0; col < CHAR_W; ++col) {
                if (glyph[row][col] == '#') {
                    voxpaint::paintVoxel(voxels,
                                          charX + col,
                                          yTop  - row,
                                          zc,
                                          cr, cg, cb);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// Per-item palette for unselected cubes (cycle through 6 1-bit colours).
// -----------------------------------------------------------------------
static void cubeColor(int i, uint8_t& r, uint8_t& g, uint8_t& b)
{
    static const uint8_t pal[6][3] = {
        {255,   0,   0},  // red
        {  0, 255,   0},  // green
        {255, 255,   0},  // yellow
        {  0,   0, 255},  // blue
        {255,   0, 255},  // magenta
        {  0, 255, 255},  // cyan
    };
    int k = ((i % 6) + 6) % 6;
    r = pal[k][0]; g = pal[k][1]; b = pal[k][2];
}

// -----------------------------------------------------------------------
// menu.json lookup: try cwd-relative first, then exe-dir-relative
// (handles being launched from project root or from build/).
// -----------------------------------------------------------------------
static std::string findMenuJson()
{
    namespace fs = std::filesystem;
    if (fs::exists("config/menu.json")) return "config/menu.json";

    char exe[4096] = {0};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        fs::path p(exe);
        fs::path candidate = p.parent_path().parent_path() / "config" / "menu.json";
        if (fs::exists(candidate)) return candidate.string();
    }
    return "config/menu.json";  // not found; loadJsonFile will print
}

int MenuApp::iconKindForId(const std::string& id) const
{
    if (id == "cube")      return ICON_CUBE;
    if (id == "torus")     return ICON_TORUS;
    if (id == "particles") return ICON_DOTS;
    if (id == "flow")      return ICON_DOTS;        // share the dots icon
    if (id == "wireframe") return ICON_TEXT;        // overridden in draw()
    if (id == "corridor")  return ICON_CORRIDOR;
    if (id == "city")      return ICON_CITY;
    if (id == "morph")     return ICON_MORPH;
    if (id == "invaders")  return ICON_PYRAMID;
    if (id == "fluid")     return ICON_DROPLET;
    if (id == "wave")      return ICON_WAVE;
    return ICON_CUBE;
}

void MenuApp::loadEntries(const std::string& /*ignored*/)
{
    entries_.clear();

    std::string path = findMenuJson();
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
        fprintf(stderr, "MenuApp: loaded %s (%zu entries)\n",
                path.c_str(), entries_.size());
    } else {
        fprintf(stderr, "MenuApp: no menu.json (tried '%s'), using defaults\n",
                path.c_str());
    }

    if (entries_.empty()) {
        entries_.push_back({"cube",      "Cube",      "",  ICON_CUBE});
        entries_.push_back({"torus",     "Torus",     "",  ICON_TORUS});
        entries_.push_back({"particles", "Particles", "",  ICON_DOTS});
    }

    if (!scanDir.empty()) {
        std::error_code ec;
        if (std::filesystem::is_directory(scanDir, ec)) {
            for (const auto& de : std::filesystem::directory_iterator(scanDir, ec)) {
                if (!de.is_regular_file()) continue;
                auto p = de.path();
                auto ext = p.extension().string();
                for (auto& ch : ext)
                    ch = (char)tolower((unsigned char)ch);
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
                ent.iconKind = ICON_TEXT;
                entries_.push_back(std::move(ent));
            }
        }
    }
}

void MenuApp::setup(Renderer& /*renderer*/)
{
    loadEntries("config/menu.json");
    carouselAngle_     = 0.0f;
    angularVel_        = 0.0f;
    thumbsUpHeld_      = 0;
    pinchPrev_         = false;
    pinchCooldown_     = 0;
    pendingLaunch_.clear();
}

int MenuApp::selectedIndex() const
{
    if (entries_.empty()) return 0;
    int N = (int)entries_.size();
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
    if (!pendingLaunch_.empty()) return;
    if (entries_.empty()) return;

    if (pinchCooldown_ > 0) --pinchCooldown_;

    if (!hand.hand_detected) {
        angularVel_ *= ANGULAR_DECAY;
        carouselAngle_ += angularVel_;
        thumbsUpHeld_ = 0;
        pinchPrev_    = false;
        return;
    }

    Gesture g = detectGesture(hand);

    // THUMBS_UP — freeze velocity, run launch hold counter.
    if (g == Gesture::THUMBS_UP) {
        angularVel_ = 0.0f;
        pinchPrev_  = false;
        thumbsUpHeld_++;
        if (thumbsUpHeld_ >= THUMBS_UP_LAUNCH_FRAMES) {
            const Entry& sel = entries_[selectedIndex()];
            if (sel.id == "wireframe" && !sel.obj.empty())
                pendingLaunch_ = "wireframe:" + sel.obj;
            else
                pendingLaunch_ = sel.id;
            fprintf(stderr, "MenuApp: launching '%s'\n", pendingLaunch_.c_str());
        }
        return;
    }
    thumbsUpHeld_ = 0;

    // PINCH (rising edge) — snap to next item, ignore finger-drag this frame.
    bool pinching = (g == Gesture::PINCH);
    if (pinching && !pinchPrev_ && pinchCooldown_ == 0) {
        int N = (int)entries_.size();
        carouselAngle_ -= 2.0f * (float)M_PI / (float)N;
        angularVel_     = 0.0f;
        pinchCooldown_  = PINCH_COOLDOWN_FRAMES;
    }
    pinchPrev_ = pinching;

    if (!pinching) {
        // Normal scroll: index-tip X drives angular velocity.
        float offset = hand.lm_x[8] - 0.5f;
        float target = -offset * SPEED_SCALE;
        angularVel_ += (target - angularVel_) * SMOOTHING;
    }
    carouselAngle_ += angularVel_;
}

// =======================================================================
//                           ICON RENDERING
// =======================================================================

// Map model-space (mx,my,mz) to voxel coords given the icon's centre
// (cx,cy,cz) (model space) and per-axis scale s (model space).
static inline void modelToVoxel(float cx, float cy, float cz, float s,
                                float mx, float my, float mz,
                                int& vx, int& vy, int& vz)
{
    vx = (int)roundf((cx + mx * s) * 0.5f * (VOXEL_W - 1) + (VOXEL_W - 1) * 0.5f);
    vy = (int)roundf((cy + my * s) * 0.5f * (VOXEL_H - 1) + (VOXEL_H - 1) * 0.5f);
    vz = (int)roundf((cz + mz * s) * 0.5f * (VOXEL_D - 1) + (VOXEL_D - 1) * 0.5f);
}

static inline void modelCenterToVoxel(float cx, float cy, float cz,
                                      int& vx, int& vy, int& vz)
{
    vx = (int)roundf(cx * 0.5f * (VOXEL_W - 1) + (VOXEL_W - 1) * 0.5f);
    vy = (int)roundf(cy * 0.5f * (VOXEL_H - 1) + (VOXEL_H - 1) * 0.5f);
    vz = (int)roundf(cz * 0.5f * (VOXEL_D - 1) + (VOXEL_D - 1) * 0.5f);
}

static void drawWireframeEdges(uint8_t* voxels, float cx, float cy, float cz, float s,
                               const float (*verts)[3], int nv,
                               const int (*edges)[2], int ne,
                               uint8_t r, uint8_t g, uint8_t b)
{
    constexpr int MAX_V = 32;
    int vx[MAX_V], vy[MAX_V], vz[MAX_V];
    int n = std::min(nv, MAX_V);
    for (int i = 0; i < n; ++i)
        modelToVoxel(cx, cy, cz, s, verts[i][0], verts[i][1], verts[i][2],
                     vx[i], vy[i], vz[i]);
    for (int i = 0; i < ne; ++i) {
        int a = edges[i][0], b2 = edges[i][1];
        if (a >= n || b2 >= n) continue;
        voxpaint::paint3DLine(voxels, vx[a], vy[a], vz[a],
                                       vx[b2], vy[b2], vz[b2], r, g, b);
    }
}

void MenuApp::drawIcon(uint8_t* voxels, int kind, float cx, float cy, float cz,
                       float s, bool /*highlighted*/) const
{
    // Selected icons render in white; the per-item colour palette is reserved
    // for the unselected cubes so the foreground stays clearly distinct.
    const uint8_t r = 255, g = 255, b = 255;

    switch (kind) {
    case ICON_CUBE: {
        static const float V[8][3] = {
            {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
            {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
        };
        static const int E[12][2] = {
            {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        drawWireframeEdges(voxels, cx, cy, cz, s, V, 8, E, 12, r, g, b);
        return;
    }
    case ICON_TORUS: {
        const int N = 36;
        const int P = 2, Q = 3;
        const float R = 0.68f, rr = 0.28f;
        int prevX = 0, prevY = 0, prevZ = 0;
        bool havePrev = false;
        for (int i = 0; i <= N; ++i) {
            float t = (float)i / (float)N * 2.0f * (float)M_PI;
            float ring = R + rr * cosf((float)Q * t);
            float x = ring * cosf((float)P * t);
            float y = 0.9f * rr * sinf((float)Q * t);
            float z = ring * sinf((float)P * t);
            int vx, vy, vz;
            modelToVoxel(cx, cy, cz, s, x, y, z, vx, vy, vz);
            if (havePrev)
                voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz, r, g, b);
            prevX = vx; prevY = vy; prevZ = vz;
            havePrev = true;
        }
        return;
    }
    case ICON_DOTS: {
        static const float P[8][3] = {
            { 0.0f, 0.0f, 0.0f}, { 0.6f, 0.2f,-0.3f},
            {-0.5f, 0.5f, 0.4f}, { 0.3f,-0.6f, 0.2f},
            {-0.3f,-0.3f,-0.6f}, { 0.7f,-0.1f, 0.6f},
            {-0.7f, 0.1f,-0.2f}, { 0.1f, 0.7f,-0.5f},
        };
        for (int i = 0; i < 8; ++i) {
            int vx, vy, vz;
            modelToVoxel(cx, cy, cz, s, P[i][0], P[i][1], P[i][2], vx, vy, vz);
            voxpaint::paintCube(voxels, vx, vy, vz, 0, r, g, b);
        }
        return;
    }
    case ICON_TETRA: {
        static const float V[4][3] = {
            { 0.0f,  1.0f,  0.0f}, {-0.9f, -0.5f, -0.5f},
            { 0.9f, -0.5f, -0.5f}, { 0.0f, -0.5f,  0.9f},
        };
        static const int E[6][2] = {{0,1},{0,2},{0,3},{1,2},{2,3},{3,1}};
        drawWireframeEdges(voxels, cx, cy, cz, s, V, 4, E, 6, r, g, b);
        return;
    }
    case ICON_CORRIDOR: {
        static const float wallL[2][3] = {{-0.7f, -1.0f, -0.5f}, {-0.7f,  0.7f, -0.5f}};
        static const float wallR[2][3] = {{ 0.7f, -1.0f, -0.5f}, { 0.7f,  0.7f, -0.5f}};
        static const float floorPts[3][3] = {
            {-0.7f, -1.0f, -0.5f}, { 0.0f, -1.0f,  0.0f}, { 0.7f, -1.0f,  0.5f}
        };
        int ax, ay, az, bx, by_, bz;
        modelToVoxel(cx, cy, cz, s, wallL[0][0], wallL[0][1], wallL[0][2], ax, ay, az);
        modelToVoxel(cx, cy, cz, s, wallL[1][0], wallL[1][1], wallL[1][2], bx, by_, bz);
        voxpaint::paint3DLine(voxels, ax, ay, az, bx, by_, bz, r, g, b);
        modelToVoxel(cx, cy, cz, s, wallR[0][0], wallR[0][1], wallR[0][2], ax, ay, az);
        modelToVoxel(cx, cy, cz, s, wallR[1][0], wallR[1][1], wallR[1][2], bx, by_, bz);
        voxpaint::paint3DLine(voxels, ax, ay, az, bx, by_, bz, r, g, b);
        int fpx[3], fpy[3], fpz[3];
        for (int i = 0; i < 3; ++i)
            modelToVoxel(cx, cy, cz, s, floorPts[i][0], floorPts[i][1], floorPts[i][2],
                         fpx[i], fpy[i], fpz[i]);
        voxpaint::paint3DLine(voxels, fpx[0], fpy[0], fpz[0], fpx[1], fpy[1], fpz[1], r, g, b);
        voxpaint::paint3DLine(voxels, fpx[1], fpy[1], fpz[1], fpx[2], fpy[2], fpz[2], r, g, b);
        return;
    }
    case ICON_CITY: {
        static const float buildings[3][4] = {
            {-0.65f, 0.0f, 0.20f, 0.5f},
            { 0.00f, 0.0f, 0.22f, 1.0f},
            { 0.65f, 0.0f, 0.18f, 0.3f},
        };
        for (int b2 = 0; b2 < 3; ++b2) {
            float bcx = buildings[b2][0], bcz = buildings[b2][1];
            float bhw = buildings[b2][2], bh  = buildings[b2][3];
            float x0m = bcx - bhw, x1m = bcx + bhw;
            float z0m = bcz - 0.18f, z1m = bcz + 0.18f;
            float yBot = -1.0f, yTop = -1.0f + bh * 2.0f;
            int px0, py0, pz0, px1, py1, pz1;
            modelToVoxel(cx, cy, cz, s, x0m, yBot, z0m, px0, py0, pz0);
            modelToVoxel(cx, cy, cz, s, x1m, yTop, z1m, px1, py1, pz1);
            voxpaint::paint3DLine(voxels, px0, py0, pz0, px0, py1, pz0, r, g, b);
            voxpaint::paint3DLine(voxels, px1, py0, pz0, px1, py1, pz0, r, g, b);
            voxpaint::paint3DLine(voxels, px1, py0, pz1, px1, py1, pz1, r, g, b);
            voxpaint::paint3DLine(voxels, px0, py0, pz1, px0, py1, pz1, r, g, b);
            voxpaint::paint3DLine(voxels, px0, py1, pz0, px1, py1, pz0, r, g, b);
            voxpaint::paint3DLine(voxels, px1, py1, pz0, px1, py1, pz1, r, g, b);
            voxpaint::paint3DLine(voxels, px1, py1, pz1, px0, py1, pz1, r, g, b);
            voxpaint::paint3DLine(voxels, px0, py1, pz1, px0, py1, pz0, r, g, b);
        }
        return;
    }
    case ICON_MORPH: {
        // Cycle three polyhedra (octahedron → cube → tetra) every ~2 s
        // off the wallclock so the icon visibly "morphs".
        using namespace std::chrono;
        int phase = (int)(steady_clock::now().time_since_epoch() / seconds(2)) % 3;
        if (phase == 0) {
            static const float V[6][3] = {
                { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0},
                { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
            };
            static const int E[12][2] = {
                {0,2},{0,3},{0,4},{0,5},{1,2},{1,3},
                {1,4},{1,5},{2,4},{2,5},{3,4},{3,5},
            };
            drawWireframeEdges(voxels, cx, cy, cz, s, V, 6, E, 12, r, g, b);
        } else if (phase == 1) {
            static const float V[8][3] = {
                {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
                {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
            };
            static const int E[12][2] = {
                {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}
            };
            drawWireframeEdges(voxels, cx, cy, cz, s, V, 8, E, 12, r, g, b);
        } else {
            static const float V[4][3] = {
                { 0.0f,  1.0f,  0.0f}, {-0.9f, -0.5f, -0.5f},
                { 0.9f, -0.5f, -0.5f}, { 0.0f, -0.5f,  0.9f},
            };
            static const int E[6][2] = {{0,1},{0,2},{0,3},{1,2},{2,3},{3,1}};
            drawWireframeEdges(voxels, cx, cy, cz, s, V, 4, E, 6, r, g, b);
        }
        return;
    }
    case ICON_PYRAMID: {
        // 4-edge wireframe pyramid, apex up — same shape as the invaders ship.
        static const float V[5][3] = {
            {-0.7f, -0.7f, -0.7f},
            { 0.7f, -0.7f, -0.7f},
            { 0.7f, -0.7f,  0.7f},
            {-0.7f, -0.7f,  0.7f},
            { 0.0f,  1.0f,  0.0f},   // apex
        };
        static const int E[8][2] = {
            {0,1},{1,2},{2,3},{3,0},   // base square
            {0,4},{1,4},{2,4},{3,4},   // four edges to apex
        };
        drawWireframeEdges(voxels, cx, cy, cz, s, V, 5, E, 8, r, g, b);
        return;
    }
    case ICON_DROPLET: {
        // 4 ribs from a top apex curving down to a rounded bottom point,
        // joined by an equator ring. Reads as a teardrop.
        const int RIB_SEGMENTS = 4;
        const int RIBS = 8;
        // Apex above, rounded bottom below.
        const float APEX_Y   = 0.95f;
        const float BOTTOM_Y = -0.7f;
        const float MID_Y    = -0.15f;
        const float MID_R    = 0.55f;
        // Equator ring + apex-to-equator + equator-to-bottom segments.
        for (int i = 0; i < RIBS; ++i) {
            float a0 = 2.0f * (float)M_PI * i       / (float)RIBS;
            float a1 = 2.0f * (float)M_PI * (i + 1) / (float)RIBS;
            int ax, ay, az, bx, by_, bz;
            // equator segment
            modelToVoxel(cx, cy, cz, s,
                         MID_R * cosf(a0), MID_Y, MID_R * sinf(a0), ax, ay, az);
            modelToVoxel(cx, cy, cz, s,
                         MID_R * cosf(a1), MID_Y, MID_R * sinf(a1), bx, by_, bz);
            voxpaint::paint3DLine(voxels, ax, ay, az, bx, by_, bz, r, g, b);

            // apex → equator (slight curve via interior point at y=0.5)
            int prevX = 0, prevY = 0, prevZ = 0;
            bool havePrev = false;
            for (int seg = 0; seg <= RIB_SEGMENTS; ++seg) {
                float t = (float)seg / RIB_SEGMENTS;
                float yy  = APEX_Y * (1.0f - t) + MID_Y * t;
                float rr2 = MID_R  * t * t;          // pinch toward apex
                int vx, vy, vz;
                modelToVoxel(cx, cy, cz, s,
                             rr2 * cosf(a0), yy, rr2 * sinf(a0), vx, vy, vz);
                if (havePrev)
                    voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz, r, g, b);
                prevX = vx; prevY = vy; prevZ = vz; havePrev = true;
            }
            // equator → bottom point (straight rounding line)
            int bxp, byp, bzp;
            modelToVoxel(cx, cy, cz, s, 0.0f, BOTTOM_Y, 0.0f, bxp, byp, bzp);
            voxpaint::paint3DLine(voxels, ax, ay, az, bxp, byp, bzp, r, g, b);
        }
        return;
    }
    case ICON_WAVE: {
        // Two cycles of a sine in the X-Y plane at z=0.
        const int N = 32;
        int prevX = 0, prevY = 0, prevZ = 0;
        bool havePrev = false;
        for (int i = 0; i <= N; ++i) {
            float x = -1.0f + 2.0f * (float)i / (float)N;
            float y = 0.5f * sinf(x * (float)M_PI * 2.0f);
            int vx, vy, vz;
            modelToVoxel(cx, cy, cz, s, x, y, 0.0f, vx, vy, vz);
            if (havePrev)
                voxpaint::paint3DLine(voxels, prevX, prevY, prevZ, vx, vy, vz, r, g, b);
            prevX = vx; prevY = vy; prevZ = vz; havePrev = true;
        }
        return;
    }
    default:
        return;
    }
}

// =======================================================================
//                              DRAW
// =======================================================================
void MenuApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    if (entries_.empty()) { renderer.uploadVoxelBuffer(voxels); return; }

    int N = (int)entries_.size();
    int sel = selectedIndex();

    for (int i = 0; i < N; ++i) {
        float a  = 2.0f * (float)M_PI * i / N + carouselAngle_;
        float cx = CAROUSEL_RADIUS * sinf(a);
        float cz = CAROUSEL_RADIUS * cosf(a);
        float cy = (i == sel) ? 0.08f : 0.0f;

        if (i == sel) {
            const Entry& ent = entries_[i];
            int vxc, vyc, vzc;
            modelCenterToVoxel(cx, cy, cz, vxc, vyc, vzc);
            if (ent.id == "wireframe") {
                drawText(voxels, ent.label, vxc, vyc, vzc, 255, 255, 255);
            } else {
                drawIcon(voxels, ent.iconKind, cx, cy, cz,
                         ICON_SIZE_HIGHLIGHT, /*highlighted=*/true);
            }
        } else {
            int vxc, vyc, vzc;
            modelCenterToVoxel(cx, cy, cz, vxc, vyc, vzc);
            uint8_t cr, cg, cb;
            cubeColor(i, cr, cg, cb);
            voxpaint::paintCube(voxels, vxc, vyc, vzc,
                                UNSELECTED_CUBE_HALF, cr, cg, cb);
        }
    }

    // Rising blue fill on the selected item's bbox while THUMBS_UP is held.
    if (thumbsUpHeld_ > 0) {
        float progress = clampf((float)thumbsUpHeld_ / (float)THUMBS_UP_LAUNCH_FRAMES,
                                0.0f, 1.0f);
        int threshold = (int)(progress * VOXEL_H);

        float a   = 2.0f * (float)M_PI * sel / N + carouselAngle_;
        float icx = CAROUSEL_RADIUS * sinf(a);
        float icy = 0.08f;
        float icz = CAROUSEL_RADIUS * cosf(a);

        int vxc, vyc, vzc;
        modelCenterToVoxel(icx, icy, icz, vxc, vyc, vzc);
        int hvx = (int)ceilf(ICON_SIZE_HIGHLIGHT * 0.5f * (VOXEL_W - 1));
        int hvy = (int)ceilf(ICON_SIZE_HIGHLIGHT * 0.5f * (VOXEL_H - 1));
        int hvz = (int)ceilf(ICON_SIZE_HIGHLIGHT * 0.5f * (VOXEL_D - 1));

        int x0 = std::max(0,        vxc - hvx), x1 = std::min(VOXEL_W - 1, vxc + hvx);
        int y0 = std::max(0,        vyc - hvy), y1 = std::min(threshold - 1, vyc + hvy);
        int z0 = std::max(0,        vzc - hvz), z1 = std::min(VOXEL_D - 1, vzc + hvz);

        for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    int idx = ((z * VOXEL_H + y) * VOXEL_W + x) * 4;
                    if (voxels[idx + 3] != 0) {
                        voxels[idx + 0] = 0;
                        voxels[idx + 1] = 80;
                        voxels[idx + 2] = 255;
                    }
                }
    }

    renderer.uploadVoxelBuffer(voxels);
}
