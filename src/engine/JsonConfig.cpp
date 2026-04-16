#include "JsonConfig.h"
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

// Simple recursive-descent parser. Tracks position in `s` via `pos`.
// Skips whitespace and // line comments between tokens.
struct Parser {
    const std::string& s;
    size_t pos = 0;
    bool   ok  = true;
    std::string err;

    Parser(const std::string& src) : s(src) {}

    void skipSpace() {
        while (pos < s.size()) {
            char c = s[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos++;
            } else if (c == '/' && pos + 1 < s.size() && s[pos + 1] == '/') {
                while (pos < s.size() && s[pos] != '\n') pos++;
            } else {
                break;
            }
        }
    }

    bool atEnd() { skipSpace(); return pos >= s.size(); }

    char peek() { skipSpace(); return pos < s.size() ? s[pos] : '\0'; }

    bool match(char c) {
        skipSpace();
        if (pos < s.size() && s[pos] == c) { pos++; return true; }
        return false;
    }

    void fail(const std::string& msg) {
        if (ok) {
            ok = false;
            err = msg + " at offset " + std::to_string(pos);
        }
    }

    JsonValue parseValue() {
        skipSpace();
        if (pos >= s.size()) { fail("unexpected end of input"); return {}; }
        char c = s[pos];
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        fail("unexpected character");
        return {};
    }

    JsonValue parseObject() {
        JsonValue v; v.type = JsonValue::Type::Object;
        if (!match('{')) { fail("expected '{'"); return v; }
        skipSpace();
        if (match('}')) return v;
        while (ok) {
            skipSpace();
            if (s[pos] != '"') { fail("expected string key"); return v; }
            JsonValue key = parseString();
            if (!ok) return v;
            if (!match(':')) { fail("expected ':'"); return v; }
            JsonValue val = parseValue();
            if (!ok) return v;
            v.objVal[key.strVal] = std::move(val);
            if (match(',')) continue;
            if (match('}')) return v;
            fail("expected ',' or '}'");
            return v;
        }
        return v;
    }

    JsonValue parseArray() {
        JsonValue v; v.type = JsonValue::Type::Array;
        if (!match('[')) { fail("expected '['"); return v; }
        skipSpace();
        if (match(']')) return v;
        while (ok) {
            JsonValue item = parseValue();
            if (!ok) return v;
            v.arrVal.push_back(std::move(item));
            if (match(',')) continue;
            if (match(']')) return v;
            fail("expected ',' or ']'");
            return v;
        }
        return v;
    }

    JsonValue parseString() {
        JsonValue v; v.type = JsonValue::Type::String;
        if (!match('"')) { fail("expected '\"'"); return v; }
        std::string out;
        while (pos < s.size() && s[pos] != '"') {
            char c = s[pos++];
            if (c == '\\' && pos < s.size()) {
                char esc = s[pos++];
                switch (esc) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += esc; break;  // permissive
                }
            } else {
                out += c;
            }
        }
        if (pos >= s.size()) { fail("unterminated string"); return v; }
        pos++;  // consume closing "
        v.strVal = std::move(out);
        return v;
    }

    JsonValue parseNumber() {
        JsonValue v; v.type = JsonValue::Type::Number;
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() &&
               (isdigit((unsigned char)s[pos]) || s[pos] == '.' ||
                s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-'))
            pos++;
        std::string num = s.substr(start, pos - start);
        v.numVal = strtod(num.c_str(), nullptr);
        return v;
    }

    JsonValue parseBool() {
        JsonValue v; v.type = JsonValue::Type::Bool;
        if (s.compare(pos, 4, "true") == 0)  { pos += 4; v.boolVal = true;  return v; }
        if (s.compare(pos, 5, "false") == 0) { pos += 5; v.boolVal = false; return v; }
        fail("invalid boolean"); return v;
    }

    JsonValue parseNull() {
        JsonValue v;
        if (s.compare(pos, 4, "null") == 0) { pos += 4; return v; }
        fail("invalid null"); return v;
    }
};

}  // namespace

bool loadJsonFile(const std::string& path, JsonValue& root)
{
    root = {};
    std::ifstream f(path);
    if (!f.good()) return false;  // silent: caller decides

    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();

    Parser p(src);
    root = p.parseValue();
    if (!p.ok) {
        fprintf(stderr, "JsonConfig: failed to parse '%s': %s\n",
                path.c_str(), p.err.c_str());
        root = {};
        return false;
    }
    return true;
}
