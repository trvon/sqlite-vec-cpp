#pragma once

#include <sqlite3.h>
#include <cstdint>
#include <span>
#include "../distances/cosine.hpp"
#include "../distances/hamming.hpp"
#include "../distances/l1.hpp"
#include "../distances/l2.hpp"
#include "../utils/error.hpp"
#include "../vector_view.hpp"
#include "context.hpp"
#include "value.hpp"

namespace sqlite_vec_cpp::sqlite {

/// Vector element type enum (matches original sqlite-vec)
enum class VectorElementType : int {
    Float32 = 223 + 0,
    Bit = 223 + 1,
    Int8 = 223 + 2,
};

/// Extract vector from sqlite3_value with type information
/// Returns span to vector data and element type
template <typename T> Result<VectorView<const T>> extract_vector_from_value(const Value& value) {
    if (value.is_null()) {
        return err<VectorView<const T>>(Error::invalid_argument("Vector value is NULL"));
    }

    if (!value.is_blob()) {
        return err<VectorView<const T>>(Error::invalid_argument("Vector value must be BLOB type"));
    }

    auto blob = value.as_blob();
    if (blob.size() % sizeof(T) != 0) {
        return err<VectorView<const T>>(
            Error::invalid_argument("Blob size not aligned to element size"));
    }

    const T* data = reinterpret_cast<const T*>(blob.data());
    std::size_t dimensions = blob.size() / sizeof(T);

    return Result<VectorView<const T>>(VectorView<const T>(data, dimensions));
}

/// Helper to determine element type from subtype
/// If subtype is 0 (no subtype), defaults to Float32 for legacy/stored blobs
inline VectorElementType get_element_type_from_subtype(int subtype) {
    // Subtypes: 223+0 = float32, 223+1 = bit, 223+2 = int8
    if (subtype == static_cast<int>(VectorElementType::Int8)) {
        return VectorElementType::Int8;
    } else if (subtype == static_cast<int>(VectorElementType::Bit)) {
        return VectorElementType::Bit;
    }
    // Default to float32 (for subtype == 223 or subtype == 0)
    // Note: subtypes are not persisted in database columns, so stored blobs have subtype 0
    return VectorElementType::Float32;
}

/// SQLite function: vec_distance_l2(a, b)
/// Calculates L2 (Euclidean) distance between two vectors
inline void vec_distance_l2_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_distance_l2 requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    // Determine element type from subtype
    // Note: Subtypes may be 0 for stored values (not persisted), so we default to Float32
    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    VectorElementType elem_type_a = get_element_type_from_subtype(subtype_a);
    VectorElementType elem_type_b = get_element_type_from_subtype(subtype_b);

    if (elem_type_a != elem_type_b) {
        context.result_error("Vector element types must match");
        return;
    }

    switch (elem_type_a) {
        case VectorElementType::Float32: {
            auto vec_a = extract_vector_from_value<float>(val_a);
            auto vec_b = extract_vector_from_value<float>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::l2_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Int8: {
            auto vec_a = extract_vector_from_value<std::int8_t>(val_a);
            auto vec_b = extract_vector_from_value<std::int8_t>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::l2_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Bit: {
            context.result_error("Cannot calculate L2 distance between bitvectors");
            break;
        }
    }
}

/// SQLite function: vec_distance_l1(a, b)
/// Calculates L1 (Manhattan) distance between two vectors
inline void vec_distance_l1_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_distance_l1 requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    VectorElementType elem_type_a = get_element_type_from_subtype(subtype_a);
    VectorElementType elem_type_b = get_element_type_from_subtype(subtype_b);

    if (elem_type_a != elem_type_b) {
        context.result_error("Vector element types must match");
        return;
    }

    switch (elem_type_a) {
        case VectorElementType::Float32: {
            auto vec_a = extract_vector_from_value<float>(val_a);
            auto vec_b = extract_vector_from_value<float>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::l1_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Int8: {
            auto vec_a = extract_vector_from_value<std::int8_t>(val_a);
            auto vec_b = extract_vector_from_value<std::int8_t>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::l1_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Bit: {
            context.result_error("Cannot calculate L1 distance between bitvectors");
            break;
        }
    }
}

/// SQLite function: vec_distance_cosine(a, b)
/// Calculates cosine distance between two vectors
inline void vec_distance_cosine_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_distance_cosine requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    VectorElementType elem_type_a = get_element_type_from_subtype(subtype_a);
    VectorElementType elem_type_b = get_element_type_from_subtype(subtype_b);

    if (elem_type_a != elem_type_b) {
        context.result_error("Vector element types must match");
        return;
    }

    switch (elem_type_a) {
        case VectorElementType::Float32: {
            auto vec_a = extract_vector_from_value<float>(val_a);
            auto vec_b = extract_vector_from_value<float>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::cosine_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Int8: {
            auto vec_a = extract_vector_from_value<std::int8_t>(val_a);
            auto vec_b = extract_vector_from_value<std::int8_t>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            float dist = distances::cosine_distance(vec_a->span(), vec_b->span());
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Bit: {
            context.result_error("Cannot calculate cosine distance between bitvectors");
            break;
        }
    }
}

/// SQLite function: vec_distance_hamming(a, b)
/// Calculates hamming distance between two bitvectors
inline void vec_distance_hamming_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_distance_hamming requires exactly 2 arguments");
        return;
    }

    Value val_a(argv[0]);
    Value val_b(argv[1]);

    int subtype_a = val_a.subtype();
    int subtype_b = val_b.subtype();

    VectorElementType elem_type_a = get_element_type_from_subtype(subtype_a);
    VectorElementType elem_type_b = get_element_type_from_subtype(subtype_b);

    if (elem_type_a != elem_type_b) {
        context.result_error("Vector element types must match");
        return;
    }

    switch (elem_type_a) {
        case VectorElementType::Bit: {
            auto vec_a = extract_vector_from_value<std::uint8_t>(val_a);
            auto vec_b = extract_vector_from_value<std::uint8_t>(val_b);

            if (!vec_a || !vec_b) {
                context.result_error(vec_a ? vec_b.error() : vec_a.error());
                return;
            }

            if (vec_a->size() != vec_b->size()) {
                context.result_error(Error::dimension_mismatch(vec_a->size(), vec_b->size()));
                return;
            }

            // For bitvectors, dimensions refers to number of bits, not bytes
            std::size_t bit_dimensions = vec_a->size() * 8; // bytes * 8
            float dist = distances::hamming_distance(vec_a->span(), vec_b->span(), bit_dimensions);
            context.result_double(static_cast<double>(dist));
            break;
        }
        case VectorElementType::Float32: {
            context.result_error("Cannot calculate hamming distance between float32 vectors");
            break;
        }
        case VectorElementType::Int8: {
            context.result_error("Cannot calculate hamming distance between int8 vectors");
            break;
        }
    }
}

// C-style wrappers for SQLite registration
extern "C" {
inline void vec_distance_l2(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_distance_l2_impl(ctx, argc, argv);
}

inline void vec_distance_l1(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_distance_l1_impl(ctx, argc, argv);
}

inline void vec_distance_cosine(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_distance_cosine_impl(ctx, argc, argv);
}

inline void vec_distance_hamming(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_distance_hamming_impl(ctx, argc, argv);
}
}

} // namespace sqlite_vec_cpp::sqlite
