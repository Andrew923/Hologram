#version 450 core
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

#define SLICE_COUNT  240
#define SLICE_W      128.0
#define SLICE_H      64.0
#define VOXEL_W      128.0
#define VOXEL_H      64.0
#define VOXEL_D      128.0
#define M_PI         3.14159265358979323846
#define MASK_RADIUS_BIAS 1.0

layout(binding = 0) uniform sampler3D uVoxelGrid;
layout(binding = 1, rgba8) writeonly uniform image2DArray uSliceOut;

uniform float uPanelOffset;
uniform float uOffsetSign;
uniform float uSweepDirection;
uniform int   uSwapSinCos;
uniform float uPhaseOffset;

void main()
{
    ivec2 coord      = ivec2(gl_GlobalInvocationID.xy);
    int   sliceIndex = int(gl_GlobalInvocationID.z);
    if (coord.x >= 128 || coord.y >= 64) return;

    float theta = (float(sliceIndex) * (2.0 * M_PI / float(SLICE_COUNT))) * uSweepDirection
                + uPhaseOffset;

    float cosT = cos(theta);
    float sinT = sin(theta);
    if (uSwapSinCos != 0) {
        float tmp = cosT;
        cosT = sinT;
        sinT = tmp;
    }

    float sliceX = float(coord.x) + 0.5;
    float sliceY = float(coord.y) + 0.5;
    float t      = sliceX - 0.5 * SLICE_W;

    float centerX = 0.5 * VOXEL_W;
    float centerZ = 0.5 * VOXEL_D;
    float offset  = uOffsetSign * uPanelOffset;

    float vox_x = centerX + offset * cosT - t * sinT;
    float vox_y = VOXEL_H - sliceY;
    float vox_z = centerZ + offset * sinT + t * cosT;

    float cx = vox_x - centerX;
    float cz = vox_z - centerZ;
    float maskRadius = abs(uPanelOffset) + MASK_RADIUS_BIAS;
    if (cx * cx + cz * cz < maskRadius * maskRadius) {
        imageStore(uSliceOut, ivec3(coord, sliceIndex), vec4(0.0));
        return;
    }

    if (vox_x < 0.5 || vox_x > (VOXEL_W - 0.5) ||
        vox_y < 0.5 || vox_y > (VOXEL_H - 0.5) ||
        vox_z < 0.5 || vox_z > (VOXEL_D - 0.5)) {
        imageStore(uSliceOut, ivec3(coord, sliceIndex), vec4(0.0));
        return;
    }

    ivec3 src = ivec3(int(floor(vox_x)), int(floor(vox_y)), int(floor(vox_z)));
    vec4 color = texelFetch(uVoxelGrid, src, 0);
    imageStore(uSliceOut, ivec3(coord, sliceIndex), color);
}
