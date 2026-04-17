#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

#define SLICE_COUNT  240
#define M_PI         3.14159265358979323846
// Physical gap between the two panels is 48 mm; for P2 (2 mm/px) that is
// 12 px per side.  Each panel sweeps a chord offset by this amount from the
// spin axis, not a diameter through the axis.
#define PANEL_OFFSET 12.0

// 3D voxel grid sampler — 128(X) × 64(Y) × 128(Z), RGBA8
layout(binding = 0) uniform sampler3D uVoxelGrid;

// 128×64×SLICE_COUNT output slice array image
layout(binding = 1, rgba8) writeonly uniform image2DArray uSliceOut;

void main()
{
    ivec2 coord      = ivec2(gl_GlobalInvocationID.xy);
    int   sliceIndex = int(gl_GlobalInvocationID.z);
    if (coord.x >= 128 || coord.y >= 64) return;

    float theta = float(sliceIndex) * (2.0 * M_PI / float(SLICE_COUNT));
    float cosT  = cos(theta);
    float sinT  = sin(theta);

    // coord.x ∈ [64,127] → Panel A: chord at +PANEL_OFFSET from spin axis.
    //   t = 0 at the innermost pixel (d = PANEL_OFFSET from centre),
    //   t = 63 at the outermost pixel (d ≈ 64 px from centre).
    // coord.x ∈ [0,63]   → Panel B: chord at −PANEL_OFFSET, mirrored.
    //   t = 0 at innermost (coord.x = 63), t = 63 at outermost (coord.x = 0).
    float vox_x, vox_z;
    if (coord.x >= 64) {
        float t = float(coord.x - 64);
        vox_x = (64.0 + PANEL_OFFSET * cosT - t * sinT) / 128.0;
        vox_z = (64.0 + PANEL_OFFSET * sinT + t * cosT) / 128.0;
    } else {
        float t = float(63 - coord.x);
        vox_x = (64.0 - PANEL_OFFSET * cosT + t * sinT) / 128.0;
        vox_z = (64.0 - PANEL_OFFSET * sinT - t * cosT) / 128.0;
    }

    float vox_y = float(coord.y) / 64.0;

    vec4 color = texture(uVoxelGrid, vec3(vox_x, vox_y, vox_z));
    imageStore(uSliceOut, ivec3(coord, sliceIndex), color);
}
