#version 450 core
// Paint app: per-voxel age decay + cursor stamp + recolor in one pass.
// Age is stored in a 128×64×128 R8 texture; voxel grid is rewritten every
// frame from age (the renderer's clearVoxels happens before draw, so we
// can't accumulate — we resynthesize colour from age each frame).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r8)    uniform image3D          uAge;
layout(binding = 1, rgba8) uniform writeonly image3D uVoxelGrid;

uniform vec3  uCursor;        // voxel-space cursor centre
uniform float uCursorRadius;  // radius in voxels (sphere brush)
uniform float uCursorActive;  // 0 or 1
uniform float uDecay;         // age decrement per frame

const vec4 C_HEAD = vec4(1.0, 1.0, 1.0, 1.0);  // white
const vec4 C_MID  = vec4(0.0, 1.0, 1.0, 1.0);  // cyan
const vec4 C_TAIL = vec4(1.0, 0.0, 1.0, 1.0);  // magenta
const vec4 C_DEEP = vec4(0.0, 0.0, 1.0, 1.0);  // blue

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 128 || c.y >= 64 || c.z >= 128) return;

    float age = imageLoad(uAge, c).r;

    bool stamped = false;
    if (uCursorActive > 0.5) {
        vec3 d = vec3(c) - uCursor;
        if (dot(d, d) <= uCursorRadius * uCursorRadius) {
            age = 1.0;
            stamped = true;
        }
    }
    if (!stamped) age = max(0.0, age - uDecay);

    imageStore(uAge, c, vec4(age));

    vec4 col;
    if      (age > 0.75) col = C_HEAD;
    else if (age > 0.50) col = C_MID;
    else if (age > 0.25) col = C_TAIL;
    else if (age > 0.05) col = C_DEEP;
    else                 col = vec4(0.0);

    imageStore(uVoxelGrid, c, col);
}
