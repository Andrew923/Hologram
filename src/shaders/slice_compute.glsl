#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

#define SLICE_COUNT  240
#define M_PI         3.14159265358979323846
// Physical gap between the two panels is 48 mm; for P2 (2 mm/px) that is
// 12 px per side.  Each panel sweeps a chord offset by this amount from the
// spin axis, not a diameter through the axis.
//
// Each slice represents one physical panel.  Both panels sample from the same
// 240-slice rotation buffer; the second panel simply reads with a 120-slice
// phase offset (180°).  The chord formula below is identical for both.
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

    // Each pixel sweeps a chord at perpendicular distance PANEL_OFFSET from
    // the spin axis.  coord.x = 64 is the midpoint of the chord (closest to
    // the spin axis, at distance PANEL_OFFSET).  t < 0 and t > 0 extend to
    // the two ends of the chord along the tangential direction (-sinT, cosT).
    float t     = float(coord.x) - 64.0;   // -64..+63
    float vox_x = (64.0 + PANEL_OFFSET * cosT - t * sinT) / 128.0;
    float vox_y = float(coord.y) / 64.0;
    float vox_z = (64.0 + PANEL_OFFSET * sinT + t * cosT) / 128.0;

    vec4 color = texture(uVoxelGrid, vec3(vox_x, vox_y, vox_z));
    imageStore(uSliceOut, ivec3(coord, sliceIndex), color);
}
