#pragma once
#include <vector>
#include <string>
#include <array>

// Minimal OBJ loader: extracts vertices and unique edges from face definitions.
// Supports 'v' (vertex) and 'f' (face) lines only. Normals, UVs, materials
// are ignored. Faces of any polygon count are decomposed into edges.

struct ObjMesh {
    std::vector<std::array<float, 3>> vertices;

    // Each edge is a pair of vertex indices (0-based).
    // Deduplicated: edge (a,b) stored with a < b.
    std::vector<std::array<int, 2>> edges;
};

// Load an OBJ file from disk. Returns true on success.
// On failure, mesh is left empty and an error is printed to stderr.
bool loadObj(const std::string& path, ObjMesh& mesh);
