#pragma once
// -----------------------------------------------------------------------
// CameraConfig — pinhole intrinsics + per-user hand reference length.
//
// Loaded from config/camera.json. If the file is absent or malformed,
// `valid` stays false and consumers should fall back to a heuristic
// depth estimator (e.g. hand-size proxy).
// -----------------------------------------------------------------------
#include <string>

struct CameraConfig {
    bool  valid = false;

    int   image_width  = 640;
    int   image_height = 480;

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    // 5-element OpenCV distortion vector (k1, k2, p1, p2, k3).
    float dist[5] = {0, 0, 0, 0, 0};

    // Physical length of the per-user reference bone (wrist → index MCP),
    // in meters. Produced by python/calibrate_hand.py.
    float user_index_bone_m = 0.09f;

    // Populate from `config/camera.json`. Returns true on success.
    // Silent failure (just sets valid=false) if the file is missing.
    bool loadFromFile(const std::string& path);
};
