#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

#define SLICE_COUNT 240
#define M_PI 3.14159265358979323846  // slice angle math constant

// 3D voxel grid sampler — 128(X) × 64(Y) × 128(Z), RGBA8
layout(binding = 0) uniform sampler3D uVoxelGrid;

// 128×64×SLICE_COUNT output slice array image
layout(binding = 1, rgba8) writeonly uniform image2DArray uSliceOut;

void main()
{
    ivec2 coord      = ivec2(gl_GlobalInvocationID.xy);
    int   sliceIndex = int(gl_GlobalInvocationID.z);
    if (coord.x >= 128 || coord.y >= 64) return;

    // Compute theta for this slice index
    float theta = float(sliceIndex) * (2.0 * M_PI / float(SLICE_COUNT));

    // coord.x ∈ [0,127] = horizontal pixel (maps to cylinder radius+offset)
    // coord.y ∈ [0,63]  = vertical pixel (maps to voxel Y)
    float r     = float(coord.x) - 64.0;       // range [-64, +63], center→edge
    float vox_x = (64.0 + r * cos(theta)) / 128.0;
    float vox_y = float(coord.y) / 64.0;
    float vox_z = (64.0 + r * sin(theta)) / 128.0;

    vec4 color = texture(uVoxelGrid, vec3(vox_x, vox_y, vox_z));
    imageStore(uSliceOut, ivec3(coord, sliceIndex), color);
}
