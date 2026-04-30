#version 450 core
// Render alive cells to the voxel grid. Each cell at grid coord c paints a
// 2×2×2 voxel block at (2c, 2c+1) — chunky enough to bridge the rim arc gap
// at 240 slices/rev. Dead cells write zero so the voxel grid is fully
// resynthesised each frame (the renderer also clears it pre-draw).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r8ui)  uniform readonly  uimage3D uCells;
layout(binding = 1, rgba8) uniform writeonly image3D  uVoxelGrid;

uniform vec4 uColor;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;

    vec4 col = imageLoad(uCells, c).r != 0u ? uColor : vec4(0.0);
    ivec3 v  = c * 2;
    imageStore(uVoxelGrid, v + ivec3(0, 0, 0), col);
    imageStore(uVoxelGrid, v + ivec3(1, 0, 0), col);
    imageStore(uVoxelGrid, v + ivec3(0, 1, 0), col);
    imageStore(uVoxelGrid, v + ivec3(1, 1, 0), col);
    imageStore(uVoxelGrid, v + ivec3(0, 0, 1), col);
    imageStore(uVoxelGrid, v + ivec3(1, 0, 1), col);
    imageStore(uVoxelGrid, v + ivec3(0, 1, 1), col);
    imageStore(uVoxelGrid, v + ivec3(1, 1, 1), col);
}
