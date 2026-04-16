#pragma once
// -----------------------------------------------------------------------
// JsonConfig — tiny, dependency-free JSON reader.
//
// Supports the subset of JSON we need:
//   - Objects { "key": value, ... }
//   - Arrays  [ value, ... ]
//   - Strings "..."
//   - Numbers (parsed as double)
//   - Booleans true/false
//   - null
//
// Robust against trailing whitespace and C-style comments (// ... EOL)
// so users can document their config files. Not a spec-compliant parser
// — it's intended for small hand-written configs, not arbitrary JSON.
// -----------------------------------------------------------------------
#include <string>
#include <vector>
#include <map>
#include <memory>

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool    boolVal = false;
    double  numVal  = 0.0;
    std::string strVal;
    std::vector<JsonValue> arrVal;
    std::map<std::string, JsonValue> objVal;

    // Safe accessors with defaults. Return defaultVal if the type
    // doesn't match or the key isn't present.
    bool        asBool  (bool d = false) const { return type == Type::Bool   ? boolVal : d; }
    double      asNumber(double d = 0.0) const { return type == Type::Number ? numVal  : d; }
    std::string asString(const std::string& d = "") const {
        return type == Type::String ? strVal : d;
    }
    const JsonValue& at(const std::string& key) const {
        static const JsonValue empty;
        if (type != Type::Object) return empty;
        auto it = objVal.find(key);
        return it == objVal.end() ? empty : it->second;
    }
    bool has(const std::string& key) const {
        return type == Type::Object && objVal.count(key) > 0;
    }
    const std::vector<JsonValue>& asArray() const {
        static const std::vector<JsonValue> empty;
        return type == Type::Array ? arrVal : empty;
    }
};

// Load a JSON file from disk. Returns true on success and populates root.
// On parse failure, returns false and prints an error to stderr; root is
// set to Null.
bool loadJsonFile(const std::string& path, JsonValue& root);
