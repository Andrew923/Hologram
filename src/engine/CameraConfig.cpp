#include "CameraConfig.h"
#include "JsonConfig.h"
#include <cstdio>

bool CameraConfig::loadFromFile(const std::string& path)
{
    JsonValue root;
    if (!loadJsonFile(path, root)) {
        valid = false;
        return false;
    }

    image_width  = (int)root.at("image_width").asNumber(image_width);
    image_height = (int)root.at("image_height").asNumber(image_height);
    fx = (float)root.at("fx").asNumber(0.0);
    fy = (float)root.at("fy").asNumber(0.0);
    cx = (float)root.at("cx").asNumber((double)image_width  * 0.5);
    cy = (float)root.at("cy").asNumber((double)image_height * 0.5);
    user_index_bone_m = (float)root.at("user_index_bone_m").asNumber(0.09);

    const auto& d = root.at("dist_coeffs").asArray();
    for (int i = 0; i < 5 && i < (int)d.size(); ++i) {
        dist[i] = (float)d[i].asNumber(0.0);
    }

    valid = (fx > 0.0f && fy > 0.0f);
    if (!valid) {
        fprintf(stderr, "CameraConfig: '%s' loaded but fx/fy missing\n",
                path.c_str());
    }
    return valid;
}
