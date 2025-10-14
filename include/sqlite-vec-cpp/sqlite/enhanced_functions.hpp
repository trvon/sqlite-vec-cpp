#pragma once

#include <sqlite3.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "context.hpp"
#include "functions.hpp"
#include "parsers.hpp"
#include "value.hpp"

namespace sqlite_vec_cpp::sqlite {

/// NEW: vec_dot(a, b) -> scalar (dot product)
/// Not in original sqlite-vec, useful for cosine similarity calculation
inline void vec_dot_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_dot requires exactly 2 arguments");
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

        // Use std::inner_product for dot product
        double result = std::inner_product(vec_a->begin(), vec_a->end(), vec_b->begin(), 0.0);

        context.result_double(result);
    } else {
        context.result_error("vec_dot only supports float32 vectors");
    }
}

/// NEW: vec_magnitude(vector) -> scalar (L2 norm)
inline void vec_magnitude_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_magnitude requires exactly 1 argument");
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

        // Calculate L2 norm using std::transform_reduce (C++17 parallel algorithm)
        double magnitude = std::sqrt(std::transform_reduce(
            vec->begin(), vec->end(), 0.0, std::plus<>{}, [](float x) { return x * x; }));

        context.result_double(magnitude);
    } else {
        context.result_error("vec_magnitude only supports float32 vectors");
    }
}

/// NEW: vec_scale(vector, scalar) -> scaled vector
inline void vec_scale_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 2) {
        context.result_error("vec_scale requires exactly 2 arguments");
        return;
    }

    Value vec_val(argv[0]);
    Value scalar_val(argv[1]);

    if (!scalar_val.is_float() && !scalar_val.is_integer()) {
        context.result_error("vec_scale scalar must be numeric");
        return;
    }

    double scalar =
        scalar_val.is_float() ? scalar_val.as_double() : static_cast<double>(scalar_val.as_int64());

    int subtype = vec_val.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (elem_type == VectorElementType::Float32) {
        auto vec = parse_vector_from_value<float>(vec_val);

        if (!vec) {
            context.result_error(vec.error());
            return;
        }

        std::vector<float> result(vec->size());
        std::transform(vec->begin(), vec->end(), result.begin(),
                       [scalar](float x) { return x * static_cast<float>(scalar); });

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_scale only supports float32 vectors");
    }
}

/// NEW: vec_mean(vector) -> scalar (arithmetic mean)
inline void vec_mean_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_mean requires exactly 1 argument");
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

        if (vec->empty()) {
            context.result_null();
            return;
        }

        double mean = std::accumulate(vec->begin(), vec->end(), 0.0) / vec->size();
        context.result_double(mean);
    } else {
        context.result_error("vec_mean only supports float32 vectors");
    }
}

/// NEW: vec_std(vector) -> scalar (standard deviation)
inline void vec_std_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_std requires exactly 1 argument");
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

        if (vec->size() < 2) {
            context.result_null();
            return;
        }

        // Calculate mean
        double mean = std::accumulate(vec->begin(), vec->end(), 0.0) / vec->size();

        // Calculate variance
        double variance = std::transform_reduce(vec->begin(), vec->end(), 0.0, std::plus<>{},
                                                [mean](float x) {
                                                    double diff = x - mean;
                                                    return diff * diff;
                                                }) /
                          (vec->size() - 1); // Sample standard deviation (N-1)

        context.result_double(std::sqrt(variance));
    } else {
        context.result_error("vec_std only supports float32 vectors");
    }
}

/// NEW: vec_min(vector) -> scalar (minimum element)
inline void vec_min_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_min requires exactly 1 argument");
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

        if (vec->empty()) {
            context.result_null();
            return;
        }

        float min_val = *std::min_element(vec->begin(), vec->end());
        context.result_double(static_cast<double>(min_val));
    } else {
        context.result_error("vec_min only supports float32 vectors");
    }
}

/// NEW: vec_max(vector) -> scalar (maximum element)
inline void vec_max_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 1) {
        context.result_error("vec_max requires exactly 1 argument");
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

        if (vec->empty()) {
            context.result_null();
            return;
        }

        float max_val = *std::max_element(vec->begin(), vec->end());
        context.result_double(static_cast<double>(max_val));
    } else {
        context.result_error("vec_max only supports float32 vectors");
    }
}

/// NEW: vec_clamp(vector, min, max) -> clamped vector
inline void vec_clamp_impl(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    Context context(ctx);

    if (argc != 3) {
        context.result_error("vec_clamp requires exactly 3 arguments");
        return;
    }

    Value vec_val(argv[0]);
    Value min_val(argv[1]);
    Value max_val(argv[2]);

    if (!min_val.is_float() && !min_val.is_integer()) {
        context.result_error("vec_clamp min must be numeric");
        return;
    }

    if (!max_val.is_float() && !max_val.is_integer()) {
        context.result_error("vec_clamp max must be numeric");
        return;
    }

    double min_d =
        min_val.is_float() ? min_val.as_double() : static_cast<double>(min_val.as_int64());
    double max_d =
        max_val.is_float() ? max_val.as_double() : static_cast<double>(max_val.as_int64());

    int subtype = vec_val.subtype();
    VectorElementType elem_type = get_element_type_from_subtype(subtype);

    if (elem_type == VectorElementType::Float32) {
        auto vec = parse_vector_from_value<float>(vec_val);

        if (!vec) {
            context.result_error(vec.error());
            return;
        }

        std::vector<float> result(vec->size());
        std::transform(vec->begin(), vec->end(), result.begin(), [min_d, max_d](float x) {
            return std::clamp(static_cast<double>(x), min_d, max_d);
        });

        auto blob_data = std::as_bytes(std::span(result));
        context.result_blob_with_subtype(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(blob_data.data()),
                                          blob_data.size()),
            static_cast<unsigned int>(VectorElementType::Float32));
    } else {
        context.result_error("vec_clamp only supports float32 vectors");
    }
}

// C-style wrappers for SQLite registration
extern "C" {
inline void vec_dot(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_dot_impl(ctx, argc, argv);
}

inline void vec_magnitude(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_magnitude_impl(ctx, argc, argv);
}

inline void vec_scale(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_scale_impl(ctx, argc, argv);
}

inline void vec_mean(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_mean_impl(ctx, argc, argv);
}

inline void vec_std(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_std_impl(ctx, argc, argv);
}

inline void vec_min(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_min_impl(ctx, argc, argv);
}

inline void vec_max(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_max_impl(ctx, argc, argv);
}

inline void vec_clamp(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    vec_clamp_impl(ctx, argc, argv);
}
}

} // namespace sqlite_vec_cpp::sqlite
