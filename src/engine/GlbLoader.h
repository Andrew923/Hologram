#pragma once
#include "ObjLoader.h"
#include <string>

// Load a GLB (binary GLTF 2.0) file.
//
// Extracts all mesh primitives from the file, combines them into a single
// ObjMesh, and, when available, reads per-vertex colors from the COLOR_0
// vertex attribute or from the material's pbrMetallicRoughness.baseColorFactor.
//
// On success returns true and populates mesh.  On failure prints a message
// to stderr and returns false with mesh left empty.
bool loadGlb(const std::string& path, ObjMesh& mesh);
