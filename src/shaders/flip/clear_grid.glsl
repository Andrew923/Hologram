#version 450 core
// Clear all per-cell grid state at the start of each FLIP frame.
// Pressure textures are zeroed on the CPU side via glClearTexImage to stay
// within the 8-image-unit hardware limit (bindings 0–7 only).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0, r32i) uniform iimage3D aVelX;
layout(binding = 1, r32i) uniform iimage3D aVelY;
layout(binding = 2, r32i) uniform iimage3D aVelZ;
layout(binding = 3, r32i) uniform iimage3D aWeight;
layout(binding = 4, r32f) uniform image3D  gVelX;
layout(binding = 5, r32f) uniform image3D  gVelY;
layout(binding = 6, r32f) uniform image3D  gVelZ;
layout(binding = 7, r8ui) uniform uimage3D gCellType;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (c.x >= 64 || c.y >= 32 || c.z >= 64) return;
    imageStore(aVelX,     c, ivec4(0));
    imageStore(aVelY,     c, ivec4(0));
    imageStore(aVelZ,     c, ivec4(0));
    imageStore(aWeight,   c, ivec4(0));
    imageStore(gVelX,     c, vec4(0.0));
    imageStore(gVelY,     c, vec4(0.0));
    imageStore(gVelZ,     c, vec4(0.0));
    imageStore(gCellType, c, uvec4(1u)); // 1 = AIR (default)
}
