#include "GlbLoader.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Minimal recursive-descent JSON parser
// ─────────────────────────────────────────────────────────────────────────────

enum class JType { Null, Bool, Number, String, Array, Object };

struct JVal {
    JType type = JType::Null;
    double num  = 0.0;
    bool   bval = false;
    std::string              str;
    std::vector<JVal>        arr;
    std::map<std::string, JVal> obj;

    bool isNull() const { return type == JType::Null; }
    bool isNum()  const { return type == JType::Number; }
    bool isStr()  const { return type == JType::String; }
    bool isArr()  const { return type == JType::Array; }
    bool isObj()  const { return type == JType::Object; }

    double asNum(double d = 0.0)  const { return isNum()  ? num  : d; }
    int    asInt(int    d = 0)    const { return isNum()  ? (int)num : d; }
    bool   asBool(bool  d = false) const { return type == JType::Bool ? bval : d; }

    const JVal& at(const std::string& k) const {
        static const JVal empty;
        if (!isObj()) return empty;
        auto it = obj.find(k);
        return it != obj.end() ? it->second : empty;
    }
    const JVal& at(size_t i) const {
        static const JVal empty;
        return (isArr() && i < arr.size()) ? arr[i] : empty;
    }
    size_t size() const { return isArr() ? arr.size() : obj.size(); }
};

static void skipWs(const char* s, size_t len, size_t& p)
{
    while (p < len &&
           (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
        ++p;
}

static JVal parseVal(const char* s, size_t len, size_t& p);

static std::string parseStr(const char* s, size_t len, size_t& p)
{
    ++p; // skip opening '"'
    std::string out;
    while (p < len && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < len) {
            ++p;
            switch (s[p]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += s[p]; break;
            }
        } else {
            out += s[p];
        }
        ++p;
    }
    if (p < len) ++p; // skip closing '"'
    return out;
}

static JVal parseVal(const char* s, size_t len, size_t& p)
{
    skipWs(s, len, p);
    if (p >= len) return {};

    JVal v;
    char c = s[p];

    if (c == '"') {
        v.type = JType::String;
        v.str  = parseStr(s, len, p);

    } else if (c == '{') {
        v.type = JType::Object;
        ++p; // skip '{'
        skipWs(s, len, p);
        while (p < len && s[p] != '}') {
            skipWs(s, len, p);
            if (p < len && s[p] == '"') {
                std::string key = parseStr(s, len, p);
                skipWs(s, len, p);
                if (p < len && s[p] == ':') ++p;
                v.obj.emplace(std::move(key), parseVal(s, len, p));
                skipWs(s, len, p);
                if (p < len && s[p] == ',') ++p;
                skipWs(s, len, p);
            } else {
                break; // malformed
            }
        }
        if (p < len && s[p] == '}') ++p;

    } else if (c == '[') {
        v.type = JType::Array;
        ++p; // skip '['
        skipWs(s, len, p);
        while (p < len && s[p] != ']') {
            v.arr.push_back(parseVal(s, len, p));
            skipWs(s, len, p);
            if (p < len && s[p] == ',') ++p;
            skipWs(s, len, p);
        }
        if (p < len && s[p] == ']') ++p;

    } else if (c == 't' && p + 3 < len && strncmp(s + p, "true",  4) == 0) {
        v.type = JType::Bool; v.bval = true;  p += 4;
    } else if (c == 'f' && p + 4 < len && strncmp(s + p, "false", 5) == 0) {
        v.type = JType::Bool; v.bval = false; p += 5;
    } else if (c == 'n' && p + 3 < len && strncmp(s + p, "null",  4) == 0) {
        p += 4; // type stays Null
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char* end = nullptr;
        v.type = JType::Number;
        v.num  = strtod(s + p, &end);
        p = static_cast<size_t>(end - s);
    }

    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLTF binary helpers
// ─────────────────────────────────────────────────────────────────────────────

// Byte size of one component value for a given GLTF componentType.
static int compSize(int ct)
{
    switch (ct) {
        case 5120: case 5121: return 1; // BYTE / UNSIGNED_BYTE
        case 5122: case 5123: return 2; // SHORT / UNSIGNED_SHORT
        case 5125: case 5126: return 4; // UNSIGNED_INT / FLOAT
        default: return 0;
    }
}

// Number of scalar components for a GLTF accessor type string.
static int numComps(const std::string& t)
{
    if (t == "SCALAR") return 1;
    if (t == "VEC2")   return 2;
    if (t == "VEC3")   return 3;
    if (t == "VEC4")   return 4;
    return 0;
}

// Read one float component from binary data, applying normalisation if needed.
static float readFloat(const uint8_t* p, int ct, bool norm)
{
    switch (ct) {
        case 5126: { float f; memcpy(&f, p, 4); return f; }
        case 5121: return norm ? p[0] / 255.0f : static_cast<float>(p[0]);
        case 5123: { uint16_t u; memcpy(&u, p, 2);
                     return norm ? u / 65535.0f : static_cast<float>(u); }
        case 5122: { int16_t  i; memcpy(&i, p, 2);
                     return norm ? std::max(i / 32767.0f, -1.0f) : static_cast<float>(i); }
        case 5120: { int8_t   i; memcpy(&i, p, 1);
                     return norm ? std::max(i / 127.0f, -1.0f) : static_cast<float>(i); }
        default: return 0.0f;
    }
}

// Read an unsigned integer index (used for the INDICES accessor).
static uint32_t readUint(const uint8_t* p, int ct)
{
    switch (ct) {
        case 5121: return p[0];
        case 5123: { uint16_t u; memcpy(&u, p, 2); return u; }
        case 5125: { uint32_t u; memcpy(&u, p, 4); return u; }
        default: return 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// loadGlb
// ─────────────────────────────────────────────────────────────────────────────

bool loadGlb(const std::string& path, ObjMesh& mesh)
{
    // ── 1. Read file into memory ──────────────────────────────────────────────
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "GlbLoader: cannot open '%s'\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    if (fsize < 12) {
        fclose(f);
        fprintf(stderr, "GlbLoader: file too small '%s'\n", path.c_str());
        return false;
    }

    std::vector<uint8_t> data(static_cast<size_t>(fsize));
    if (fread(data.data(), 1, static_cast<size_t>(fsize), f) !=
            static_cast<size_t>(fsize)) {
        fclose(f);
        fprintf(stderr, "GlbLoader: read error '%s'\n", path.c_str());
        return false;
    }
    fclose(f);

    // ── 2. Parse GLB header ───────────────────────────────────────────────────
    uint32_t magic = 0, version = 0;
    memcpy(&magic,   data.data() + 0, 4);
    memcpy(&version, data.data() + 4, 4);

    if (magic != 0x46546C67u) {
        fprintf(stderr, "GlbLoader: not a GLB file '%s'\n", path.c_str());
        return false;
    }
    if (version != 2) {
        fprintf(stderr, "GlbLoader: unsupported GLB version %u\n", version);
        return false;
    }

    // ── 3. Parse GLB chunks ───────────────────────────────────────────────────
    const uint8_t* jsonChunk = nullptr;
    uint32_t       jsonLen   = 0;
    const uint8_t* binChunk  = nullptr;
    uint32_t       binLen    = 0;

    size_t pos = 12;
    while (pos + 8 <= data.size()) {
        uint32_t chunkLen = 0, chunkType = 0;
        memcpy(&chunkLen,  data.data() + pos,     4);
        memcpy(&chunkType, data.data() + pos + 4, 4);
        pos += 8;

        if (pos + chunkLen > data.size()) break; // truncated

        if (chunkType == 0x4E4F534Au) {      // 'JSON'
            jsonChunk = data.data() + pos;
            jsonLen   = chunkLen;
        } else if (chunkType == 0x004E4942u) { // 'BIN\0'
            binChunk = data.data() + pos;
            binLen   = chunkLen;
        }
        pos += chunkLen;
    }

    if (!jsonChunk || jsonLen == 0) {
        fprintf(stderr, "GlbLoader: no JSON chunk in '%s'\n", path.c_str());
        return false;
    }

    // ── 4. Parse GLTF JSON ────────────────────────────────────────────────────
    size_t jpos = 0;
    JVal root = parseVal(reinterpret_cast<const char*>(jsonChunk), jsonLen, jpos);
    if (root.isNull()) {
        fprintf(stderr, "GlbLoader: JSON parse failed for '%s'\n", path.c_str());
        return false;
    }

    const JVal& accessorsArr = root.at("accessors");
    const JVal& bufViewsArr  = root.at("bufferViews");
    const JVal& meshesArr    = root.at("meshes");
    const JVal& materialsArr = root.at("materials");

    if (!accessorsArr.isArr() || !bufViewsArr.isArr() || !meshesArr.isArr()) {
        fprintf(stderr, "GlbLoader: missing required GLTF arrays in '%s'\n",
                path.c_str());
        return false;
    }

    // Helper: resolve an accessor to a data pointer + per-element stride.
    // Returns nullptr on error (invalid indices, unsupported buffer, etc.).
    // Out-parameters: compType, nComps, normalized, count, stride.
    auto resolveAccessor = [&](int accIdx,
                                int& compType,
                                int& nComps,
                                bool& normalized,
                                int& count,
                                int& stride) -> const uint8_t*
    {
        if (accIdx < 0 || static_cast<size_t>(accIdx) >= accessorsArr.size())
            return nullptr;

        const JVal& acc = accessorsArr.at(static_cast<size_t>(accIdx));
        int bvIdx      = acc.at("bufferView").asInt(-1);
        compType       = acc.at("componentType").asInt(0);
        count          = acc.at("count").asInt(0);
        normalized     = acc.at("normalized").asBool(false);
        nComps         = numComps(acc.at("type").isStr() ? acc.at("type").str : "");
        int accByteOff = acc.at("byteOffset").asInt(0);

        if (bvIdx < 0 || static_cast<size_t>(bvIdx) >= bufViewsArr.size())
            return nullptr;

        const JVal& bv = bufViewsArr.at(static_cast<size_t>(bvIdx));
        int bufIdx     = bv.at("buffer").asInt(0);
        int bvByteOff  = bv.at("byteOffset").asInt(0);
        int bvStride   = bv.at("byteStride").asInt(0);

        // Only buffer 0 (the GLB BIN chunk) is supported.
        if (bufIdx != 0 || !binChunk) return nullptr;

        int offset = bvByteOff + accByteOff;
        if (offset < 0 || static_cast<uint32_t>(offset) >= binLen)
            return nullptr;

        int elemSize = nComps * compSize(compType);
        stride = (bvStride > 0) ? bvStride : elemSize;

        // Bounds check: last element must fit inside the BIN chunk.
        if (count > 0) {
            size_t lastByte = static_cast<size_t>(offset) +
                              static_cast<size_t>(count - 1) * static_cast<size_t>(stride) +
                              static_cast<size_t>(elemSize);
            if (lastByte > binLen) return nullptr;
        }

        return binChunk + offset;
    };

    // ── 5. Process all meshes ─────────────────────────────────────────────────
    mesh.vertices.clear();
    mesh.edges.clear();
    mesh.colors.clear();

    std::set<std::pair<int,int>> edgeSet;
    bool anyColors = false; // true if at least one primitive provided color data

    for (size_t mi = 0; mi < meshesArr.size(); ++mi) {
        const JVal& meshJ     = meshesArr.at(mi);
        const JVal& primsArr  = meshJ.at("primitives");
        if (!primsArr.isArr()) continue;

        for (size_t pi = 0; pi < primsArr.size(); ++pi) {
            const JVal& prim  = primsArr.at(pi);
            const JVal& attrs = prim.at("attributes");

            int posAccIdx = attrs.at("POSITION").asInt(-1);
            int colAccIdx = attrs.at("COLOR_0").asInt(-1);
            int idxAccIdx = prim.at("indices").asInt(-1);
            int matIdx    = prim.at("material").asInt(-1);

            if (posAccIdx < 0) continue;

            // ── Read positions ────────────────────────────────────────────────
            int  posCompType, posNComps, posCount, posStride;
            bool posNorm;
            const uint8_t* posData = resolveAccessor(
                posAccIdx, posCompType, posNComps, posNorm, posCount, posStride);
            if (!posData || posNComps < 3) continue;

            int vertexOffset = static_cast<int>(mesh.vertices.size());
            for (int vi = 0; vi < posCount; ++vi) {
                const uint8_t* elem = posData + static_cast<size_t>(vi) * posStride;
                int cs = compSize(posCompType);
                float x = readFloat(elem + 0 * cs, posCompType, posNorm);
                float y = readFloat(elem + 1 * cs, posCompType, posNorm);
                float z = readFloat(elem + 2 * cs, posCompType, posNorm);
                mesh.vertices.push_back({x, y, z});
            }

            // ── Read vertex colors ────────────────────────────────────────────
            bool primHasColors = false;

            if (colAccIdx >= 0) {
                int  colCompType, colNComps, colCount, colStride;
                bool colNorm;
                const uint8_t* colData = resolveAccessor(
                    colAccIdx, colCompType, colNComps, colNorm, colCount, colStride);
                if (colData && colNComps >= 3 && colCount == posCount) {
                    primHasColors = true;
                    anyColors = true;
                    // Pad any previously-added vertices (other primitives without
                    // color) with the default cyan so the vectors stay aligned.
                    while (static_cast<int>(mesh.colors.size()) < vertexOffset)
                        mesh.colors.push_back({0, 255, 255});

                    int cs = compSize(colCompType);
                    for (int vi = 0; vi < colCount; ++vi) {
                        const uint8_t* elem = colData + static_cast<size_t>(vi) * colStride;
                        float rf = readFloat(elem + 0 * cs, colCompType, colNorm);
                        float gf = readFloat(elem + 1 * cs, colCompType, colNorm);
                        float bf = readFloat(elem + 2 * cs, colCompType, colNorm);
                        auto clamp01 = [](float v){ return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
                        mesh.colors.push_back({
                            static_cast<uint8_t>(clamp01(rf) * 255.0f),
                            static_cast<uint8_t>(clamp01(gf) * 255.0f),
                            static_cast<uint8_t>(clamp01(bf) * 255.0f)
                        });
                    }
                }
            }

            // ── Fallback: material base color ─────────────────────────────────
            if (!primHasColors && matIdx >= 0 && materialsArr.isArr() &&
                    static_cast<size_t>(matIdx) < materialsArr.size()) {
                const JVal& mat = materialsArr.at(static_cast<size_t>(matIdx));
                const JVal& bcf = mat.at("pbrMetallicRoughness").at("baseColorFactor");
                if (bcf.isArr() && bcf.size() >= 3) {
                    auto clamp01 = [](float v){ return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
                    uint8_t r = static_cast<uint8_t>(clamp01(static_cast<float>(bcf.at(0).asNum(1.0))) * 255.0f);
                    uint8_t g = static_cast<uint8_t>(clamp01(static_cast<float>(bcf.at(1).asNum(1.0))) * 255.0f);
                    uint8_t b = static_cast<uint8_t>(clamp01(static_cast<float>(bcf.at(2).asNum(1.0))) * 255.0f);

                    primHasColors = true;
                    anyColors = true;
                    while (static_cast<int>(mesh.colors.size()) < vertexOffset)
                        mesh.colors.push_back({0, 255, 255});
                    for (int vi = 0; vi < posCount; ++vi)
                        mesh.colors.push_back({r, g, b});
                }
            }

            // If this primitive has no color but others did, pad with default.
            if (!primHasColors && anyColors) {
                while (static_cast<int>(mesh.colors.size()) <
                       static_cast<int>(mesh.vertices.size()))
                    mesh.colors.push_back({0, 255, 255});
            }

            // ── Build edges from face indices ─────────────────────────────────
            auto addEdge = [&](int u, int v) {
                if (u > v) std::swap(u, v);
                edgeSet.insert({u, v});
            };

            if (idxAccIdx >= 0) {
                int  idxCompType, idxNComps, idxCount, idxStride;
                bool idxNorm;
                const uint8_t* idxData = resolveAccessor(
                    idxAccIdx, idxCompType, idxNComps, idxNorm, idxCount, idxStride);
                if (idxData && idxNComps == 1) {
                    for (int ii = 0; ii + 2 < idxCount; ii += 3) {
                        int a = static_cast<int>(readUint(idxData + static_cast<size_t>(ii + 0) * idxStride, idxCompType)) + vertexOffset;
                        int b = static_cast<int>(readUint(idxData + static_cast<size_t>(ii + 1) * idxStride, idxCompType)) + vertexOffset;
                        int c = static_cast<int>(readUint(idxData + static_cast<size_t>(ii + 2) * idxStride, idxCompType)) + vertexOffset;
                        addEdge(a, b);
                        addEdge(b, c);
                        addEdge(a, c);
                    }
                }
            } else {
                // Non-indexed geometry: treat consecutive triplets as triangles.
                for (int ii = 0; ii + 2 < posCount; ii += 3) {
                    int a = vertexOffset + ii + 0;
                    int b = vertexOffset + ii + 1;
                    int c = vertexOffset + ii + 2;
                    addEdge(a, b);
                    addEdge(b, c);
                    addEdge(a, c);
                }
            }
        }
    }

    // ── 6. Finalise ───────────────────────────────────────────────────────────
    // Ensure colors vector is either empty (no color data at all) or exactly
    // the same length as vertices (pad any trailing gap with default cyan).
    if (!mesh.colors.empty()) {
        while (mesh.colors.size() < mesh.vertices.size())
            mesh.colors.push_back({0, 255, 255});
        if (mesh.colors.size() != mesh.vertices.size())
            mesh.colors.clear(); // inconsistent — disable colors
    }

    mesh.edges.reserve(edgeSet.size());
    for (auto& [a, b] : edgeSet)
        mesh.edges.push_back({a, b});

    fprintf(stderr,
            "GlbLoader: loaded '%s' — %zu vertices, %zu edges, %s colors\n",
            path.c_str(), mesh.vertices.size(), mesh.edges.size(),
            mesh.colors.empty() ? "no" : "with");

    return !mesh.vertices.empty();
}
