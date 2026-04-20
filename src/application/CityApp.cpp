#include "CityApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"
#include <cmath>
#include <cstring>

// -----------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------
static constexpr float CAM_SCALE   = 200.0f;  // finger [0,1] → world [0, 200]
static constexpr float SMOOTHING   = 0.08f;

static constexpr int   BLOCK_SIZE  = 14;   // world units per grid cell
static constexpr int   STREET_GAP  = 3;    // voxels of gap on each side of block

// -----------------------------------------------------------------------
// Deterministic building properties from grid coordinates
// -----------------------------------------------------------------------
static unsigned cityHash(int gx, int gz)
{
    unsigned h = (unsigned)(gx * 7919 + gz * 6271);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

// Returns building height [6, 52]
static float buildingHeight(int gx, int gz)
{
    unsigned h = cityHash(gx, gz);
    return 6.0f + (float)(h & 0xFFFFu) / 65535.0f * 46.0f;
}

// Returns building type 0-3
static int buildingType(int gx, int gz)
{
    unsigned h = (unsigned)(gx * 3517 + gz * 4999);
    h ^= h >> 16;
    h *= 0x9e3779b9u;
    h ^= h >> 16;
    return (int)(h & 3u);
}

// -----------------------------------------------------------------------
// Helper: draw a wireframe box given min/max voxel coordinates
// -----------------------------------------------------------------------
static void drawBox(uint8_t* voxels,
                    int x0, int y0, int z0,
                    int x1, int y1, int z1,
                    uint8_t r, uint8_t g, uint8_t b)
{
    // Bottom face
    voxpaint::paint3DLine(voxels, x0, y0, z0, x1, y0, z0, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y0, z0, x1, y0, z1, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y0, z1, x0, y0, z1, r, g, b);
    voxpaint::paint3DLine(voxels, x0, y0, z1, x0, y0, z0, r, g, b);
    // Top face
    voxpaint::paint3DLine(voxels, x0, y1, z0, x1, y1, z0, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y1, z0, x1, y1, z1, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y1, z1, x0, y1, z1, r, g, b);
    voxpaint::paint3DLine(voxels, x0, y1, z1, x0, y1, z0, r, g, b);
    // Vertical pillars
    voxpaint::paint3DLine(voxels, x0, y0, z0, x0, y1, z0, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y0, z0, x1, y1, z0, r, g, b);
    voxpaint::paint3DLine(voxels, x1, y0, z1, x1, y1, z1, r, g, b);
    voxpaint::paint3DLine(voxels, x0, y0, z1, x0, y1, z1, r, g, b);
}

// -----------------------------------------------------------------------
// Draw one building given its grid cell and camera offset
// -----------------------------------------------------------------------
static void drawBuilding(uint8_t* voxels, int gx, int gz, float camX, float camZ)
{
    float height = buildingHeight(gx, gz);
    int   type   = buildingType(gx, gz);

    // Base footprint world coordinates
    float worldX0_base = gx * BLOCK_SIZE + STREET_GAP;
    float worldZ0_base = gz * BLOCK_SIZE + STREET_GAP;
    float worldX1_base = (gx + 1) * BLOCK_SIZE - STREET_GAP;
    float worldZ1_base = (gz + 1) * BLOCK_SIZE - STREET_GAP;

    // Convert to voxel space
    auto toVX = [&](float wx) { return (int)(wx - camX + VOXEL_W * 0.5f + 0.5f); };
    auto toVZ = [&](float wz) { return (int)(wz - camZ + VOXEL_D * 0.5f + 0.5f); };

    int vy_top = (int)height;

    switch (type) {
    case 0: {
        // Tall tower — standard footprint, full height
        // Color: pale cyan-white, brighter for taller
        uint8_t bright = (uint8_t)(120 + (int)(height / 52.0f * 100));
        drawBox(voxels,
                toVX(worldX0_base), 0, toVZ(worldZ0_base),
                toVX(worldX1_base), vy_top, toVZ(worldZ1_base),
                bright, bright, 255);
        break;
    }
    case 1: {
        // Low slab — wider footprint, short height (6–14)
        float h = 6.0f + (height - 6.0f) * (8.0f / 46.0f);  // remap to [6,14]
        int   vyS = (int)h;
        // Expand footprint by 3 voxels per side
        float pad = 3.0f;
        drawBox(voxels,
                toVX(worldX0_base - pad), 0, toVZ(worldZ0_base - pad),
                toVX(worldX1_base + pad), vyS, toVZ(worldZ1_base + pad),
                255, 180, 40);   // amber
        break;
    }
    case 2: {
        // Stepped tower — full base up to half height, smaller upper box
        int halfH = vy_top / 2;
        // Lower section: full footprint
        drawBox(voxels,
                toVX(worldX0_base), 0, toVZ(worldZ0_base),
                toVX(worldX1_base), halfH, toVZ(worldZ1_base),
                180, 220, 160);  // pale green

        // Upper section: inset by 2 voxels each side
        float inset = 2.0f;
        if (halfH < vy_top) {
            drawBox(voxels,
                    toVX(worldX0_base + inset), halfH, toVZ(worldZ0_base + inset),
                    toVX(worldX1_base - inset), vy_top, toVZ(worldZ1_base - inset),
                    220, 255, 180);  // lighter green
        }
        break;
    }
    case 3: {
        // Thin spire — narrow footprint (half-width), tall
        float hw   = (worldX1_base - worldX0_base) * 0.5f;
        float midX = (worldX0_base + worldX1_base) * 0.5f;
        float midZ = (worldZ0_base + worldZ1_base) * 0.5f;
        float h    = 30.0f + (height - 6.0f) * (22.0f / 46.0f); // remap to [30,52]
        drawBox(voxels,
                toVX(midX - hw * 0.5f), 0, toVZ(midZ - hw * 0.5f),
                toVX(midX + hw * 0.5f), (int)h, toVZ(midZ + hw * 0.5f),
                180, 220, 255);  // pale blue-white
        break;
    }
    }
}

// -----------------------------------------------------------------------
// IApplication interface
// -----------------------------------------------------------------------
void CityApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
    camX_ = 100.0f;
    camZ_ = 100.0f;
}

void CityApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);
    if (!hand.hand_detected) return;

    float targetX = hand.lm_x[8] * CAM_SCALE;
    float targetZ = hand.lm_y[8] * CAM_SCALE;

    camX_ += (targetX - camX_) * SMOOTHING;
    camZ_ += (targetZ - camZ_) * SMOOTHING;
}

void CityApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    memset(voxels, 0, sizeof(voxels));

    // Determine grid range visible in the current viewport
    float viewX0 = camX_ - VOXEL_W * 0.5f;
    float viewX1 = camX_ + VOXEL_W * 0.5f;
    float viewZ0 = camZ_ - VOXEL_D * 0.5f;
    float viewZ1 = camZ_ + VOXEL_D * 0.5f;

    int gxMin = (int)floorf(viewX0 / BLOCK_SIZE) - 1;
    int gxMax = (int)ceilf( viewX1 / BLOCK_SIZE) + 1;
    int gzMin = (int)floorf(viewZ0 / BLOCK_SIZE) - 1;
    int gzMax = (int)ceilf( viewZ1 / BLOCK_SIZE) + 1;

    for (int gz = gzMin; gz <= gzMax; ++gz)
        for (int gx = gxMin; gx <= gxMax; ++gx)
            drawBuilding(voxels, gx, gz, camX_, camZ_);

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}
