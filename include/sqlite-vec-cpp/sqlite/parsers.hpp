#pragma once

#include <sqlite3.h>
#include <cctype>
#include <charconv>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include "../utils/error.hpp"
#include "../vector_view.hpp"
#include "functions.hpp"

namespace sqlite_vec_cpp::sqlite {

/// JSON whitespace detection (from sqlite-vec.c)
constexpr bool is_json_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/// Parse JSON array string to vector
/// Format: "[1.0, 2.5, 3.14, ...]"
template <typename T> Result<std::vector<T>> parse_json_array(std::string_view json_str) {
    std::vector<T> result;

    // Skip leading whitespace
    std::size_t i = 0;
    while (i < json_str.size() && is_json_space(json_str[i])) {
        ++i;
    }

    // Check if we have content
    if (i >= json_str.size()) {
        return err<std::vector<T>>(Error::invalid_argument("Empty JSON string"));
    }

    // Must start with '['
    if (json_str[i] != '[') {
        return err<std::vector<T>>(Error::invalid_argument("JSON array must start with '['"));
    }
    ++i;

    // Pre-allocate assuming reasonable size
    result.reserve(json_str.size() / 4);

    while (i < json_str.size()) {
        // Skip whitespace
        while (i < json_str.size() && is_json_space(json_str[i])) {
            ++i;
        }

        if (i >= json_str.size()) {
            return err<std::vector<T>>(Error::invalid_argument("Unexpected end of JSON array"));
        }

        // Check for end of array
        if (json_str[i] == ']') {
            break;
        }

        // Parse number
        T value{};
        const char* start = &json_str[i];
        const char* end = start + (json_str.size() - i);

        if constexpr (std::is_floating_point_v<T>) {
            // Use strtod for floating point
            char* endptr = nullptr;
            double parsed = std::strtod(start, &endptr);

            if (endptr == start) {
                return err<std::vector<T>>(Error::invalid_argument(
                    "Failed to parse number in JSON array at position " + std::to_string(i)));
            }

            value = static_cast<T>(parsed);
            i += (endptr - start);
        } else {
            // Use from_chars for integers (C++17)
            auto [ptr, ec] = std::from_chars(start, end, value);

            if (ec != std::errc{}) {
                return err<std::vector<T>>(Error::invalid_argument(
                    "Failed to parse integer in JSON array at position " + std::to_string(i)));
            }

            i += (ptr - start);
        }

        result.push_back(value);

        // Skip whitespace after number
        while (i < json_str.size() && is_json_space(json_str[i])) {
            ++i;
        }

        // Check for comma or end
        if (i < json_str.size()) {
            if (json_str[i] == ',') {
                ++i;
            } else if (json_str[i] != ']') {
                return err<std::vector<T>>(Error::invalid_argument(
                    "Expected ',' or ']' in JSON array at position " + std::to_string(i)));
            }
        }
    }

    if (result.empty()) {
        return err<std::vector<T>>(
            Error::invalid_argument("Zero-length vectors are not supported"));
    }

    return Result<std::vector<T>>(std::move(result));
}

/// Parse vector from SQLite value (BLOB or TEXT)
/// Handles both binary blobs and JSON arrays
template <typename T> Result<std::vector<T>> parse_vector_from_value(const Value& value) {
    if (value.is_null()) {
        return err<std::vector<T>>(Error::invalid_argument("Vector value is NULL"));
    }

    if (value.is_blob()) {
        // Binary format
        auto blob = value.as_blob();

        if (blob.empty()) {
            return err<std::vector<T>>(
                Error::invalid_argument("Zero-length vectors are not supported"));
        }

        if (blob.size() % sizeof(T) != 0) {
            return err<std::vector<T>>(Error::invalid_argument(
                "Invalid vector BLOB length. Must be divisible by " + std::to_string(sizeof(T)) +
                ", found " + std::to_string(blob.size())));
        }

        std::size_t num_elements = blob.size() / sizeof(T);
        std::vector<T> result(num_elements);
        std::memcpy(result.data(), blob.data(), blob.size());

        return Result<std::vector<T>>(std::move(result));
    } else if (value.is_text()) {
        // JSON format
        auto text = value.as_text();

        if (text.empty()) {
            return err<std::vector<T>>(
                Error::invalid_argument("Zero-length vectors are not supported"));
        }

        return parse_json_array<T>(text);
    }

    return err<std::vector<T>>(Error::invalid_argument("Vector value must be BLOB or TEXT type"));
}

/// Parse bitvector from value (for hamming distance)
inline Result<std::vector<std::uint8_t>>
parse_bitvector_from_value(const Value& value, std::size_t& out_bit_dimensions) {
    if (!value.is_blob()) {
        return err<std::vector<std::uint8_t>>(
            Error::invalid_argument("Bitvector value must be BLOB type"));
    }

    auto blob = value.as_blob();

    if (blob.empty()) {
        return err<std::vector<std::uint8_t>>(
            Error::invalid_argument("Zero-length bitvectors are not supported"));
    }

    out_bit_dimensions = blob.size() * CHAR_BIT;

    std::vector<std::uint8_t> result(blob.begin(), blob.end());
    return Result<std::vector<std::uint8_t>>(std::move(result));
}

/// Format vector as JSON array string
template <typename T> std::string format_vector_as_json(std::span<const T> vec) {
    if (vec.empty()) {
        return "[]";
    }

    std::string result = "[";

    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }

        if constexpr (std::is_floating_point_v<T>) {
            // Use reasonable precision for floats
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.6g", static_cast<double>(vec[i]));
            result += buffer;
        } else {
            result += std::to_string(vec[i]);
        }
    }

    result += "]";
    return result;
}

} // namespace sqlite_vec_cpp::sqlite
