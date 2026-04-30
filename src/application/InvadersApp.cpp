#include "InvadersApp.h"
#include "VoxelPaint.h"
#include "../engine/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void InvadersApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();
    resetGame();
    firstFrame_ = true;
}

void InvadersApp::resetGame()
{
    bullets_.clear();
    debris_.clear();
    powerups_.clear();
    currentWeapon_     = Weapon::NORMAL;
    weaponTimer_       = 0.0f;
    laserBeamTimer_    = 0.0f;
    fireCooldown_      = 0.0f;
    debrisSpawnTimer_  = 0.6f;     // first debris ~half interval in
    powerupSpawnTimer_ = 6.0f;     // first powerup later
    lives_             = STARTING_LIVES;
    gameOver_          = false;
    invulnTimer_       = 0.0f;
    gameOverFlashTimer_= 0.0f;
    shipX_             = CENTER_X + 30.0f;
    shipZ_             = CENTER_Z;
    pinchPrev_         = false;
}

void InvadersApp::clampShipToAnnulus(float& x, float& z) const
{
    float dx = x - CENTER_X;
    float dz = z - CENTER_Z;
    float r2 = dx * dx + dz * dz;
    if (r2 < ANNULUS_INNER * ANNULUS_INNER) {
        float r = std::sqrt(r2);
        if (r < 1e-3f) { dx = ANNULUS_INNER; dz = 0.0f; }
        else            { float s = ANNULUS_INNER / r; dx *= s; dz *= s; }
    } else if (r2 > ANNULUS_OUTER * ANNULUS_OUTER) {
        float r = std::sqrt(r2);
        float s = ANNULUS_OUTER / r;
        dx *= s; dz *= s;
    }
    x = CENTER_X + dx;
    z = CENTER_Z + dz;
}

void InvadersApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    // Time step (clamped — same idiom as FluidApp)
    auto now = std::chrono::steady_clock::now();
    float dt;
    if (firstFrame_) { dt = 1.0f / 60.0f; firstFrame_ = false; }
    else             { dt = std::chrono::duration<float>(now - lastTick_).count(); }
    lastTick_ = now;
    dt = clampf(dt, 1.0f / 120.0f, 1.0f / 30.0f);

    // -------- Read hand --------
    fingerActive_ = hand.hand_detected;
    bool pinching = false;
    if (hand.hand_detected) {
        float lx = clampf(hand.lm_x[8], 0.0f, 1.0f);
        float ly = clampf(hand.lm_y[8], 0.0f, 1.0f);
        float tgtX = lx * (float)VOXEL_W;
        float tgtZ = ly * (float)VOXEL_D;
        // Smoothing toward target — ~70% per frame, so big-but-not-twitchy
        shipX_ += (tgtX - shipX_) * 0.35f;
        shipZ_ += (tgtZ - shipZ_) * 0.35f;
        clampShipToAnnulus(shipX_, shipZ_);

        float pinchDist = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                                      hand.lm_y[4] - hand.lm_y[8]);
        pinching = (pinchDist < 0.06f);
    }
    bool pinchEdge = pinching && !pinchPrev_;
    pinchPrev_   = pinching;
    pinchActive_ = pinching;

    // -------- Game-over branch --------
    if (gameOver_) {
        gameOverFlashTimer_ += dt;
        if (pinchEdge) resetGame();
        return;
    }

    // -------- Fire weapon (edge OR cooldown-tick on hold) --------
    fireCooldown_ -= dt;
    if (pinching && (pinchEdge || fireCooldown_ <= 0.0f)) {
        fireWeapon();
        fireCooldown_ = FIRE_COOLDOWN;
    }

    // -------- Tick durations --------
    if (laserBeamTimer_ > 0.0f) laserBeamTimer_ -= dt;
    if (invulnTimer_    > 0.0f) invulnTimer_    -= dt;
    if (currentWeapon_ != Weapon::NORMAL) {
        weaponTimer_ -= dt;
        if (weaponTimer_ <= 0.0f) {
            currentWeapon_ = Weapon::NORMAL;
            weaponTimer_   = 0.0f;
        }
    }

    // -------- Move bullets --------
    for (auto& b : bullets_) {
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        b.z += b.vz * dt;
    }

    // -------- Move debris + tick hit-flash --------
    for (auto& d : debris_) {
        d.y -= DEBRIS_SPEED * dt;
        if (d.hitFlash > 0.0f) d.hitFlash -= dt;
    }

    // -------- Move powerups --------
    for (auto& p : powerups_) {
        p.y -= POWERUP_SPEED * dt;
    }

    // -------- Bullet ↔ debris collisions --------
    for (auto& b : bullets_) {
        if (b.y < 0.0f) continue;  // already removed marker
        for (auto& d : debris_) {
            if (d.hp <= 0) continue;
            float dx = b.x - d.x;
            float dy = b.y - d.y;
            float dz = b.z - d.z;
            float r  = (float)(d.hp + 2);    // hitbox grows with HP
            if (dx*dx + dy*dy + dz*dz < r*r) {
                d.hp--;
                d.hitFlash = HIT_FLASH_DURATION;
                b.y = -1.0f;                  // mark bullet for removal
                break;
            }
        }
    }

    // -------- Active laser beam: instakill anything in the ship's column --------
    if (laserBeamTimer_ > 0.0f) {
        for (auto& d : debris_) {
            if (d.hp <= 0) continue;
            float dx = d.x - shipX_;
            float dz = d.z - shipZ_;
            if (dx*dx + dz*dz < 9.0f) d.hp = 0;
        }
    }

    // -------- Ship ↔ debris collision (debris reaching the floor near ship) --------
    for (const auto& d : debris_) {
        if (d.hp <= 0) continue;
        if (d.y > SHIP_FLOOR_Y + SHIP_HEIGHT) continue;
        float dx = d.x - shipX_;
        float dz = d.z - shipZ_;
        if (dx*dx + dz*dz < 25.0f && invulnTimer_ <= 0.0f) {
            lives_--;
            invulnTimer_ = INVULN_DURATION;
            if (lives_ <= 0) gameOver_ = true;
            break;
        }
    }

    // -------- Powerup pickup / floor-miss --------
    powerups_.erase(std::remove_if(powerups_.begin(), powerups_.end(),
        [&](const Powerup& p) {
            float dx = p.x - shipX_;
            float dz = p.z - shipZ_;
            // Pickup if powerup descended to ship level + close horizontally.
            if (p.y < SHIP_FLOOR_Y + SHIP_HEIGHT && dx*dx + dz*dz < 25.0f) {
                currentWeapon_ = p.type;
                weaponTimer_   = POWERUP_DURATION;
                return true;
            }
            return p.y < SHIP_FLOOR_Y - 4.0f;     // missed it
        }), powerups_.end());

    // -------- Cull dead/escaped bullets and debris --------
    bullets_.erase(std::remove_if(bullets_.begin(), bullets_.end(),
        [](const Bullet& b) {
            return b.y < 0.0f || b.y > CEILING_Y + 2.0f
                || b.x < 0.0f || b.x >= (float)VOXEL_W
                || b.z < 0.0f || b.z >= (float)VOXEL_D;
        }), bullets_.end());

    debris_.erase(std::remove_if(debris_.begin(), debris_.end(),
        [](const Debris& d) {
            return d.hp <= 0 || d.y < SHIP_FLOOR_Y - 6.0f;
        }), debris_.end());

    // -------- Spawn timers --------
    debrisSpawnTimer_  -= dt;
    powerupSpawnTimer_ -= dt;
    if (debrisSpawnTimer_ <= 0.0f) {
        spawnDebris();
        debrisSpawnTimer_ = DEBRIS_SPAWN_INTERVAL;
    }
    if (powerupSpawnTimer_ <= 0.0f) {
        spawnPowerup();
        powerupSpawnTimer_ = POWERUP_SPAWN_INTERVAL;
    }
}

void InvadersApp::fireWeapon()
{
    float ay = SHIP_FLOOR_Y + SHIP_HEIGHT;     // bullets emerge from apex

    auto launch = [&](float vx, float vz) {
        Bullet b;
        b.x = shipX_; b.y = ay; b.z = shipZ_;
        b.vx = vx; b.vy = BULLET_SPEED; b.vz = vz;
        bullets_.push_back(b);
    };

    if (currentWeapon_ == Weapon::LASER) {
        // Beam handled in update + drawLaserBeam — no projectile.
        laserBeamTimer_ = LASER_BEAM_DURATION;
    } else if (currentWeapon_ == Weapon::SHOTGUN) {
        // Centre + 4 corner shots in a small cone.
        const float spread = BULLET_SPEED * 0.20f;
        launch(0.0f,    0.0f);
        launch( spread,  spread);
        launch( spread, -spread);
        launch(-spread,  spread);
        launch(-spread, -spread);
    } else {
        launch(0.0f, 0.0f);
    }
}

void InvadersApp::spawnDebris()
{
    if ((int)debris_.size() >= MAX_DEBRIS) return;
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    int hp;
    float r = u01(rng_);
    if      (r < 0.60f) hp = 1;
    else if (r < 0.90f) hp = 2;
    else                hp = 3;

    float ang = u01(rng_) * 2.0f * (float)M_PI;
    float rad = ANNULUS_INNER + 4.0f
              + u01(rng_) * (ANNULUS_OUTER - ANNULUS_INNER - 8.0f);

    Debris d;
    d.x = CENTER_X + rad * std::cos(ang);
    d.z = CENTER_Z + rad * std::sin(ang);
    d.y = CEILING_Y + (float)hp;     // big debris start a touch higher
    d.hp = hp;
    d.hitFlash = 0.0f;
    debris_.push_back(d);
}

void InvadersApp::spawnPowerup()
{
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    Powerup p;
    p.type = (u01(rng_) < 0.5f) ? Weapon::LASER : Weapon::SHOTGUN;
    float ang = u01(rng_) * 2.0f * (float)M_PI;
    float rad = ANNULUS_INNER + 4.0f
              + u01(rng_) * (ANNULUS_OUTER - ANNULUS_INNER - 8.0f);
    p.x = CENTER_X + rad * std::cos(ang);
    p.z = CENTER_Z + rad * std::sin(ang);
    p.y = CEILING_Y;
    powerups_.push_back(p);
}

// ============================================================
//                          RENDER
// ============================================================
void InvadersApp::draw(Renderer& renderer)
{
    static uint8_t voxels[VOXEL_BYTES];
    std::memset(voxels, 0, sizeof(voxels));

    drawShip(voxels);
    for (const auto& b : bullets_)  drawBullet(voxels, b);
    for (const auto& d : debris_)   drawDebris(voxels, d);
    for (const auto& p : powerups_) drawPowerup(voxels, p);
    if (laserBeamTimer_ > 0.0f) drawLaserBeam(voxels);
    drawLives(voxels);
    if (currentWeapon_ != Weapon::NORMAL) drawWeaponHUD(voxels);

    menuWatcher_.drawLoadingIndicator(voxels);
    renderer.uploadVoxelBuffer(voxels);
}

void InvadersApp::drawShip(uint8_t* voxels)
{
    int sx = (int)std::round(shipX_);
    int sz = (int)std::round(shipZ_);
    int sy = (int)SHIP_FLOOR_Y;
    int ay = sy + (int)SHIP_HEIGHT;

    // Default: white. Game over: blink red. Invuln: blink white/off.
    uint8_t r = 255, g = 255, b = 255;
    bool draw = true;
    if (gameOver_) {
        int phase = (int)(gameOverFlashTimer_ * 2.0f) & 1;
        if (phase) draw = false;
        else { r = 255; g = 0; b = 0; }
    } else if (invulnTimer_ > 0.0f) {
        int phase = (int)(invulnTimer_ * 10.0f) & 1;
        if (phase) draw = false;
    }
    if (!draw) return;

    int half = (int)SHIP_BASE_HALF;
    int cx[4] = { sx - half, sx + half, sx + half, sx - half };
    int cz[4] = { sz - half, sz - half, sz + half, sz + half };

    // Square base
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        voxpaint::paint3DLine(voxels, cx[i], sy, cz[i], cx[j], sy, cz[j], r, g, b);
    }
    // 4 edges to apex
    for (int i = 0; i < 4; ++i) {
        voxpaint::paint3DLine(voxels, cx[i], sy, cz[i], sx, ay, sz, r, g, b);
    }
}

void InvadersApp::drawBullet(uint8_t* voxels, const Bullet& b)
{
    voxpaint::paintCube(voxels, (int)b.x, (int)b.y, (int)b.z, 1, 0, 255, 255);
}

void InvadersApp::drawDebris(uint8_t* voxels, const Debris& d)
{
    uint8_t r = 0, g = 0, bl = 0;
    if (d.hitFlash > 0.0f) {
        r = g = bl = 255;                   // white flash on damage
    } else {
        switch (d.hp) {
            case 1: g = 255; break;          // green
            case 2: r = g = 255; break;      // yellow
            default: r = 255; break;         // red (hp 3)
        }
    }

    int dx = (int)d.x, dy = (int)d.y, dz = (int)d.z;
    int half = d.hp + 1;     // hp=1→half=2 (4³), hp=2→3 (6³), hp=3→4 (8³)

    int cx[8] = { dx-half, dx+half, dx+half, dx-half,
                  dx-half, dx+half, dx+half, dx-half };
    int cy[8] = { dy-half, dy-half, dy-half, dy-half,
                  dy+half, dy+half, dy+half, dy+half };
    int cz[8] = { dz-half, dz-half, dz+half, dz+half,
                  dz-half, dz-half, dz+half, dz+half };
    static const int E[12][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    for (int e = 0; e < 12; ++e) {
        int a = E[e][0], b = E[e][1];
        voxpaint::paint3DLine(voxels, cx[a], cy[a], cz[a],
                                       cx[b], cy[b], cz[b], r, g, bl);
    }
}

void InvadersApp::drawPowerup(uint8_t* voxels, const Powerup& p)
{
    int px = (int)p.x, py = (int)p.y, pz = (int)p.z;
    if (p.type == Weapon::LASER) {
        // Tall thin pillar — looks like a laser beam.
        for (int dy = -3; dy <= 3; ++dy)
            voxpaint::paintVoxel(voxels, px, py + dy, pz, 255, 255, 255);
    } else {
        // Wide flat disc — looks like a shotgun spread.
        for (int dx = -2; dx <= 2; ++dx)
            for (int dz = -2; dz <= 2; ++dz)
                if (dx*dx + dz*dz <= 4)
                    voxpaint::paintVoxel(voxels, px + dx, py, pz + dz,
                                          255, 255, 255);
    }
}

void InvadersApp::drawLaserBeam(uint8_t* voxels)
{
    int sx = (int)shipX_;
    int sz = (int)shipZ_;
    int sy = (int)(SHIP_FLOOR_Y + SHIP_HEIGHT);
    for (int y = sy; y <= (int)CEILING_Y; ++y) {
        for (int dx = -1; dx <= 1; ++dx)
            for (int dz = -1; dz <= 1; ++dz)
                voxpaint::paintVoxel(voxels, sx + dx, y, sz + dz, 0, 255, 255);
    }
}

void InvadersApp::drawLives(uint8_t* voxels)
{
    int sx = (int)shipX_;
    int sz = (int)shipZ_;
    int by = (int)(SHIP_FLOOR_Y + SHIP_HEIGHT + 3);
    for (int i = 0; i < lives_; ++i) {
        voxpaint::paintVoxel(voxels, sx + (i - 1) * 2, by, sz, 0, 255, 255);
    }
}

void InvadersApp::drawWeaponHUD(uint8_t* voxels)
{
    int sx = (int)shipX_;
    int sz = (int)shipZ_;
    int hy = (int)(SHIP_FLOOR_Y + SHIP_HEIGHT + 6);
    if (currentWeapon_ == Weapon::LASER) {
        // Mini pillar
        for (int dy = 0; dy < 3; ++dy)
            voxpaint::paintVoxel(voxels, sx, hy + dy, sz, 255, 255, 255);
    } else if (currentWeapon_ == Weapon::SHOTGUN) {
        // Mini horizontal bar
        for (int dx = -2; dx <= 2; ++dx)
            voxpaint::paintVoxel(voxels, sx + dx, hy, sz, 255, 255, 255);
    }
}

void InvadersApp::teardown(Renderer& /*renderer*/)
{
    bullets_.clear();
    debris_.clear();
    powerups_.clear();
}
