#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

#define SLICE_COUNT  240
#define SLICE_W      128.0
#define SLICE_H      64.0
#define VOXEL_W      128.0
#define VOXEL_H      64.0
#define VOXEL_D      128.0
#define M_PI         3.14159265358979323846
// Physical gap between the two panels is 48 mm; for P2 (2 mm/px) that is
// 12 px per side.  Each panel sweeps a chord offset by this amount from the
// spin axis, not a diameter through the axis.
#define PANEL_OFFSET 12.0
#define MASK_RADIUS  (PANEL_OFFSET + 1.0)

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

    // Use texel-center coordinates: texel i center is at i + 0.5.
    float sliceX = float(coord.x) + 0.5;
    float sliceY = float(coord.y) + 0.5;

    float t     = sliceX - 0.5 * SLICE_W;  // -63.5..+63.5
    float vox_x = 0.5 * VOXEL_W + PANEL_OFFSET * cosT - t * sinT;
    float vox_y = VOXEL_H - sliceY;
    float vox_z = 0.5 * VOXEL_D + PANEL_OFFSET * sinT + t * cosT;

    // Blank the center cylinder — panels can never illuminate within PANEL_OFFSET of the spin axis
    float cx = vox_x - 0.5 * VOXEL_W;
    float cz = vox_z - 0.5 * VOXEL_D;
    if (cx * cx + cz * cz < MASK_RADIUS * MASK_RADIUS) {
        imageStore(uSliceOut, ivec3(coord, sliceIndex), vec4(0.0));
        return;
    }

    if (vox_x < 0.5 || vox_x > (VOXEL_W - 0.5) ||
        vox_y < 0.5 || vox_y > (VOXEL_H - 0.5) ||
        vox_z < 0.5 || vox_z > (VOXEL_D - 0.5)) {
        imageStore(uSliceOut, ivec3(coord, sliceIndex), vec4(0.0));
        return;
    }

    int ix = int(floor(vox_x));
    int iy = int(floor(vox_y));
    int iz = int(floor(vox_z));

    vec4 color = texelFetch(uVoxelGrid, ivec3(ix, iy, iz), 0);
    imageStore(uSliceOut, ivec3(coord, sliceIndex), color);
}
