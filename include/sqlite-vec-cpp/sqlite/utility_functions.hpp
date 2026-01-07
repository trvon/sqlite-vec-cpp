#pragma once

#include <sqlite3.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "../distances/l2.hpp"
#include "context.hpp"
#include "functions.hpp"
#include "parsers.hpp"
#include "value.hpp"

namespace sqlite_vec_cpp::sqlite {

/// SQLite function: vec_f32(json_or_blob) -> float32 vector blob
/// Simple version WITHOUT subtype for vec0 virtual table compatibility
inline void vec_f32_simple(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_f32_simple requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    auto result = parse_vector_from_value<float>(value);

    if (!result) {
        context.result_error(result.error());
        return;
    }

    const auto& vec = result.value();
    const void* data = vec.data();
    std::size_t size = vec.size() * sizeof(float);

    // Return WITHOUT subtype so vec0 xUpdate can access the blob
    context.result_blob(
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data), size));
}

/// SQLite function: vec_f32(json_or_blob) -> float32 vector blob
inline void vec_f32_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_f32 requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    auto result = parse_vector_from_value<float>(value);

    if (!result) {
        context.result_error(result.error());
        return;
    }

    const auto& vec = result.value();

    // Create blob from vector data
    const void* data = vec.data();
    std::size_t size = vec.size() * sizeof(float);
    auto subtype = static_cast<unsigned int>(VectorElementType::Float32);

    // Use the wrapper method that correctly sets blob + subtype
    context.result_blob_with_subtype(
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data), size), subtype);
}

/// SQLite function: vec_int8(json_or_blob) -> int8 vector blob
inline void vec_int8_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_int8 requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    auto result = parse_vector_from_value<std::int8_t>(value);

    if (!result) {
        context.result_error(result.error());
        return;
    }

    const auto& vec = result.value();
    const void* data = vec.data();
    std::size_t size = vec.size() * sizeof(std::int8_t);
    auto subtype = static_cast<unsigned int>(VectorElementType::Int8);

    context.result_blob_with_subtype(
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data), size), subtype);
}

/// SQLite function: vec_bit(blob) -> bit vector blob
inline void vec_bit_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_bit requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    std::size_t bit_dimensions;
    auto result = parse_bitvector_from_value(value, bit_dimensions);

    if (!result) {
        context.result_error(result.error());
        return;
    }

    const auto& vec = result.value();
    const void* data = vec.data();
    std::size_t size = vec.size();
    auto subtype = static_cast<unsigned int>(VectorElementType::Bit);

    context.result_blob_with_subtype(
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data), size), subtype);
}

/// SQLite function: vec_length(vector) -> integer
inline void vec_length_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_length requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    int subtype = value.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (value.is_blob()) {
        auto blob = value.as_blob();
        std::size_t dimensions = 0;

        switch (elem_type) {
            case VectorElementType::Float32:
                dimensions = blob.size() / sizeof(float);
                break;
            case VectorElementType::Int8:
                dimensions = blob.size() / sizeof(std::int8_t);
                break;
            case VectorElementType::Bit:
                dimensions = blob.size() * CHAR_BIT;
                break;
        }

        context.result_int64(static_cast<std::int64_t>(dimensions));
    } else {
        context.result_error("vec_length requires a vector blob");
    }
}

/// SQLite function: vec_type(vector) -> string
inline void vec_type_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_type requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    int subtype = value.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    switch (elem_type) {
        case VectorElementType::Float32:
            context.result_text_static("float32");
            break;
        case VectorElementType::Int8:
            context.result_text_static("int8");
            break;
        case VectorElementType::Bit:
            context.result_text_static("bit");
            break;
    }
}

/// SQLite function: vec_to_json(vector) -> JSON string
inline void vec_to_json_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_to_json requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    int subtype = value.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (!value.is_blob()) {
        context.result_error("vec_to_json requires a vector blob");
        return;
    }

    switch (elem_type) {
        case VectorElementType::Float32: {
            auto vec = extract_vector_from_value<float>(value);
            if (!vec) {
                context.result_error(vec.error());
                return;
            }
            std::string json = format_vector_as_json(vec->span());
            context.result_text(json);
            break;
        }
        case VectorElementType::Int8: {
            auto vec = extract_vector_from_value<std::int8_t>(value);
            if (!vec) {
                context.result_error(vec.error());
                return;
            }
            std::string json = format_vector_as_json(vec->span());
            context.result_text(json);
            break;
        }
        case VectorElementType::Bit: {
            // For bitvectors, output as array of 0s and 1s
            auto blob = value.as_blob();
            std::string json = "[";
            for (std::size_t byte_idx = 0; byte_idx < blob.size(); ++byte_idx) {
                for (int bit_idx = 0; bit_idx < CHAR_BIT; ++bit_idx) {
                    if (byte_idx > 0 || bit_idx > 0) {
                        json += ", ";
                    }
                    json += ((blob[byte_idx] >> bit_idx) & 1) ? "1" : "0";
                }
            }
            json += "]";
            context.result_text(json);
            break;
        }
    }
}

/// SQLite function: vec_add(a, b) -> vector (element-wise addition)
inline void vec_add_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_add requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    if (subtype_a != subtype_b) {
        context.result_error("Vector element types must match");
        return;
    }

    VectorElementType elem_type = get_element_type_from_subtype(subtype_a);

    if (elem_type == VectorElementType::Float32) {
        auto vec_a = parse_vector_from_value<float>(val_a);
        auto vec_b = parse_vector_from_value<float>(val_b);

        if (!vec_a || !vec_b) {
            context.result_error(vec_a ? vec_b.error() : vec_a.error());
            return;
        }

        if (vec_a->size() != vec_b->size()) {
            context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
            return;
        }

        std::vector<float> result(vec_a->size());
        for (std::size_t i = 0; i < vec_a->size(); ++i) {
            result[i] = (*vec_a)[i] + (*vec_b)[i];
        }

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_add only supports float32 vectors");
    }
}

/// SQLite function: vec_sub(a, b) -> vector (element-wise subtraction)
inline void vec_sub_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_sub requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    if (subtype_a != subtype_b) {
        context.result_error("Vector element types must match");
        return;
    }

    VectorElementType elem_type = get_element_type_from_subtype(subtype_a);

    if (elem_type == VectorElementType::Float32) {
        auto vec_a = parse_vector_from_value<float>(val_a);
        auto vec_b = parse_vector_from_value<float>(val_b);

        if (!vec_a || !vec_b) {
            context.result_error(vec_a ? vec_b.error() : vec_a.error());
            return;
        }

        if (vec_a->size() != vec_b->size()) {
            context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
            return;
        }

        std::vector<float> result(vec_a->size());
        for (std::size_t i = 0; i < vec_a->size(); ++i) {
            result[i] = (*vec_a)[i] - (*vec_b)[i];
        }

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_sub only supports float32 vectors");
    }
}

/// SQLite function: vec_normalize(vector) -> normalized vector (L2 norm = 1)
inline void vec_normalize_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_normalize requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    int subtype = value.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (elem_type == VectorElementType::Float32) {
        auto vec = parse_vector_from_value<float>(value);

        if (!vec) {
            context.result_error(vec.error());
            return;
        }

        // Calculate L2 norm
        float norm = std::transform_reduce(vec->begin(), vec->end(), 0.0f, std::plus<float>(),
                                           [](float v) { return v * v; });
        norm = std::sqrt(norm);

        if (norm == 0.0f) {
            context.result_error("Cannot normalize zero vector");
            return;
        }

        // Normalize
        std::vector<float> result(vec->size());
        for (std::size_t i = 0; i < vec->size(); ++i) {
            result[i] = (*vec)[i] / norm;
        }

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_normalize only supports float32 vectors");
    }
}

/// SQLite function: vec_slice(vector, start, end) -> sliced vector
inline void vec_slice_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 3) {
        context.result_error("vec_slice requires exactly 3 arguments");
        return;
    }

    Value vec_val(argv[0]);
    Value start_val(argv[1]);
    Value end_val(argv[2]);

    if (!start_val.is_integer() || !end_val.is_integer()) {
        context.result_error("vec_slice start and end must be integers");
        return;
    }

    std::int64_t start = start_val.as_int64();
    std::int64_t end = end_val.as_int64();

    int subtype = vec_val.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (elem_type == VectorElementType::Float32) {
        auto vec = parse_vector_from_value<float>(vec_val);

        if (!vec) {
            context.result_error(vec.error());
            return;
        }

        std::int64_t size = static_cast<std::int64_t>(vec->size());

        // Handle negative indices (Python-style)
        if (start < 0) {
            start += size;
        }
        if (end < 0) {
            end += size;
        }

        // Clamp to valid range
        start = std::max(std::int64_t(0), std::min(start, size));
        end = std::max(std::int64_t(0), std::min(end, size));

        if (start >= end) {
            context.result_error("vec_slice start must be less than end");
            return;
        }

        std::vector<float> result(vec->begin() + start, vec->begin() + end);

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_slice only supports float32 vectors");
    }
}

/// SQLite function: vec_quantize_binary(vector) -> bit vector (quantize to binary)
/// Converts float32 vector to bitvector: values > 0 become 1, <= 0 become 0
inline void vec_quantize_binary_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_quantize_binary requires exactly 1 argument");
        return;
    }

    Value value(argv[0]);
    int subtype = value.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (elem_type != VectorElementType::Float32) {
        context.result_error("vec_quantize_binary only supports float32 vectors");
        return;
    }

    auto vec = parse_vector_from_value<float>(value);

    if (!vec) {
        context.result_error(vec.error());
        return;
    }

    // Check dimensions are divisible by 8
    if (vec->size() % CHAR_BIT != 0) {
        context.result_error("vec_quantize_binary requires dimensions divisible by " +
                             std::to_string(CHAR_BIT));
        return;
    }

    // Quantize to binary
    std::size_t num_bytes = vec->size() / CHAR_BIT;
    std::vector<std::uint8_t> result(num_bytes, 0);

    for (std::size_t i = 0; i < vec->size(); ++i) {
        if ((*vec)[i] > 0.0f) {
            std::size_t byte_idx = i / CHAR_BIT;
            std::size_t bit_idx = i % CHAR_BIT;
            result[byte_idx] |= (1 << bit_idx);
        }
    }

    auto blob_data = std::span<const std::uint8_t>(result);
    context.result_blob_with_subtype(blob_data, static_cast<unsigned int>(VectorElementType::Bit));
}

// C-style wrappers for SQLite registration
extern "C" {
inline void vec_f32(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_f32_impl(ctx, argc, argv);
}

inline void vec_int8(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_int8_impl(ctx, argc, argv);
}

inline void vec_bit(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_bit_impl(ctx, argc, argv);
}

inline void vec_length(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_length_impl(ctx, argc, argv);
}

inline void vec_type(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_type_impl(ctx, argc, argv);
}

inline void vec_to_json(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_to_json_impl(ctx, argc, argv);
}

inline void vec_add(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_add_impl(ctx, argc, argv);
}

inline void vec_sub(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_sub_impl(ctx, argc, argv);
}

inline void vec_normalize(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_normalize_impl(ctx, argc, argv);
}

inline void vec_slice(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_slice_impl(ctx, argc, argv);
}

inline void vec_quantize_binary(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_quantize_binary_impl(ctx, argc, argv);
}
}

} // namespace sqlite_vec_cpp::sqlite
