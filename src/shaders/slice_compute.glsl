#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

// 3D voxel grid sampler — 128(X) × 64(Y) × 128(Z), RGBA8
layout(binding = 0) uniform sampler3D uVoxelGrid;

// 128×64 output slice image
layout(binding = 1, rgba8) writeonly uniform image2D uSliceOut;

// Current slice angle in radians [0, 2π)
layout(location = 0) uniform float uTheta;

void main()
{
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= 128 || coord.y >= 64) return;

    // coord.x ∈ [0,127] = horizontal pixel (maps to cylinder radius+offset)
    // coord.y ∈ [0,63]  = vertical pixel (maps to voxel Y)
    float r     = float(coord.x) - 64.0;       // range [-64, +63], center→edge
    float vox_x = (64.0 + r * cos(uTheta)) / 128.0;
    float vox_y = float(coord.y) / 64.0;
    float vox_z = (64.0 + r * sin(uTheta)) / 128.0;

    vec4 color = texture(uVoxelGrid, vec3(vox_x, vox_y, vox_z));
    imageStore(uSliceOut, coord, color);
}
