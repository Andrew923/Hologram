#pragma once
#include "IApplication.h"
#include "ReturnToMenuWatcher.h"
#include <epoxy/gl.h>
#include <vector>
#include <chrono>
#include <random>
#include <cstdint>

// Space-Invaders-ish shooter on the bowl floor.
//
// Ship:    4-edge wireframe pyramid, sits on the floor (apex pointing up),
//          slides over the floor with the index-fingertip position.
// Fire:    pinch to launch a cyan bullet upward; holding pinch auto-fires
//          at FIRE_COOLDOWN cadence so the ~500 ms display lag stays
//          tolerable.
// Debris:  wireframe cubes falling from the ceiling, sized + coloured by
//          HP (1=green 4³, 2=yellow 6³, 3=red 8³). HP-many bullet hits to
//          destroy. White hit-flash for one frame on non-killing hits.
// Hits:    debris reaching the floor near the ship costs one of three
//          lives; ~1 s invulnerability + flash after a hit.
// Powerups: rare white shapes that fall down. Pillar = laser (instant
//          cyan column up the volume on pinch), disc = shotgun (5 bullets
//          in a cross). Pickup overrides previous mod and runs for
//          POWERUP_DURATION seconds. A small white HUD above the ship
//          indicates the active mod.
// Game over: ship blinks red, all spawning frozen, pinch to restart.
class InvadersApp : public IApplication {
public:
    InvadersApp() = default;
    ~InvadersApp() override = default;

    void setup(Renderer&)                   override;
    void update(const SharedHandData& hand) override;
    void draw(Renderer& renderer)           override;
    void teardown(Renderer&)                override;
    bool bypassSlicer() const               override { return false; }

    const char* requestedApp() const override {
        return menuWatcher_.shouldReturn() ? "menu" : nullptr;
    }

    // ---- Tunables -------------------------------------------------------
    static constexpr int   STARTING_LIVES = 3;
    static constexpr int   MAX_DEBRIS     = 25;

    static constexpr float CENTER_X       = 63.5f;
    static constexpr float CENTER_Z       = 63.5f;
    static constexpr float ANNULUS_INNER  = 18.0f;     // ship & spawn radius
    static constexpr float ANNULUS_OUTER  = 58.0f;
    static constexpr float SHIP_FLOOR_Y   = 8.0f;
    static constexpr float SHIP_HEIGHT    = 6.0f;
    static constexpr float SHIP_BASE_HALF = 3.0f;
    static constexpr float CEILING_Y      = 60.0f;

    static constexpr float BULLET_SPEED   = 80.0f;     // voxels/sec
    static constexpr float DEBRIS_SPEED   = 8.0f;
    static constexpr float POWERUP_SPEED  = 4.5f;
    static constexpr float DEBRIS_SPAWN_INTERVAL  = 1.2f;
    static constexpr float POWERUP_SPAWN_INTERVAL = 12.0f;
    static constexpr float POWERUP_DURATION       = 8.0f;
    static constexpr float HIT_FLASH_DURATION     = 0.10f;
    static constexpr float INVULN_DURATION        = 1.0f;
    static constexpr float FIRE_COOLDOWN          = 0.25f;
    static constexpr float LASER_BEAM_DURATION    = 0.18f;

    enum class Weapon { NORMAL, LASER, SHOTGUN };

private:
    struct Bullet  { float x, y, z; float vx, vy, vz; };
    struct Debris  { float x, y, z; int hp; float hitFlash; };
    struct Powerup { float x, y, z; Weapon type; };

    void resetGame();
    void spawnDebris();
    void spawnPowerup();
    void fireWeapon();
    void clampShipToAnnulus(float& x, float& z) const;

    // Voxel painting helpers (operate on the shared CPU voxel buffer)
    void drawShip       (uint8_t* voxels);
    void drawDebris     (uint8_t* voxels, const Debris& d);
    void drawBullet     (uint8_t* voxels, const Bullet& b);
    void drawPowerup    (uint8_t* voxels, const Powerup& p);
    void drawLaserBeam  (uint8_t* voxels);
    void drawLives      (uint8_t* voxels);
    void drawWeaponHUD  (uint8_t* voxels);

    ReturnToMenuWatcher menuWatcher_;

    // Hand state
    bool  fingerActive_ = false;
    bool  pinchActive_  = false;
    float shipX_        = CENTER_X + 30.0f;
    float shipZ_        = CENTER_Z;

    // Entities
    std::vector<Bullet>  bullets_;
    std::vector<Debris>  debris_;
    std::vector<Powerup> powerups_;

    // Powerup state
    Weapon currentWeapon_   = Weapon::NORMAL;
    float  weaponTimer_     = 0.0f;
    float  laserBeamTimer_  = 0.0f;
    float  fireCooldown_    = 0.0f;

    // Spawn timers
    float  debrisSpawnTimer_  = 0.0f;
    float  powerupSpawnTimer_ = 0.0f;

    // Game state
    int   lives_              = STARTING_LIVES;
    bool  gameOver_           = false;
    float invulnTimer_        = 0.0f;
    float gameOverFlashTimer_ = 0.0f;
    bool  pinchPrev_          = false;

    std::mt19937 rng_{0xDEADBEEF};

    std::chrono::steady_clock::time_point lastTick_;
    bool firstFrame_ = true;
};
