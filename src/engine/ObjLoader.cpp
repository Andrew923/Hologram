#include "ObjLoader.h"
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>

bool loadObj(const std::string& path, ObjMesh& mesh)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "ObjLoader: failed to open '%s'\n", path.c_str());
        return false;
    }

    mesh.vertices.clear();
    mesh.edges.clear();

    // Use a set to deduplicate edges: store (min, max) vertex index pairs
    std::set<std::pair<int,int>> edgeSet;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        if (line[0] == 'v' && line[1] == ' ') {
            // Vertex line: "v x y z"
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                mesh.vertices.push_back({x, y, z});
            }
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            // Face line: "f v1 v2 v3 ..." or "f v1/vt1/vn1 v2/vt2/vn2 ..."
            // Extract vertex indices, ignoring texture/normal indices
            std::vector<int> faceVerts;
            std::istringstream iss(line + 2);
            std::string token;

            while (iss >> token) {
                // Parse first integer before any '/'
                int vi = 0;
                if (sscanf(token.c_str(), "%d", &vi) == 1) {
                    // OBJ indices are 1-based; convert to 0-based
                    // Negative indices are relative to current vertex count
                    if (vi < 0)
                        vi = (int)mesh.vertices.size() + vi;
                    else
                        vi -= 1;

                    if (vi >= 0)
                        faceVerts.push_back(vi);
                }
            }

            // Extract edges from the face polygon
            int n = (int)faceVerts.size();
            for (int i = 0; i < n; ++i) {
                int a = faceVerts[i];
                int b = faceVerts[(i + 1) % n];
                if (a > b) std::swap(a, b);
                edgeSet.insert({a, b});
            }
        }
    }

    fclose(f);

    // Copy deduplicated edges into the mesh
    mesh.edges.reserve(edgeSet.size());
    for (auto& [a, b] : edgeSet) {
        mesh.edges.push_back({a, b});
    }

    fprintf(stderr, "ObjLoader: loaded '%s' — %zu vertices, %zu edges\n",
            path.c_str(), mesh.vertices.size(), mesh.edges.size());
    return !mesh.vertices.empty();
}
