#pragma once

#include <climits>
#include <cstdint>
#include <string>
#include <string_view>
#include "../utils/error.hpp"
#include "functions.hpp" // for VectorElementType

namespace sqlite_vec_cpp::sqlite {

/// Distance metric type for vec0 virtual tables
enum class DistanceMetric {
    L2 = 1,
    Cosine = 2,
    L1 = 3,
};

// Forward declaration
[[nodiscard]] std::size_t vector_byte_size(VectorElementType element_type,
                                           std::size_t dimensions) noexcept;

/// Vector column definition (modern C++ version)
struct VectorColumnDefinition {
    std::string name;
    std::size_t dimensions;
    VectorElementType element_type;
    DistanceMetric distance_metric;

    VectorColumnDefinition() = default; // members intentionally uninitialized

    VectorColumnDefinition(std::string_view col_name, std::size_t dims, VectorElementType elem_type,
                           DistanceMetric metric)
        : name(col_name), dimensions(dims), element_type(elem_type), distance_metric(metric) {}

    /// Get byte size for this vector column
    [[nodiscard]] std::size_t byte_size() const noexcept {
        return vector_byte_size(element_type, dimensions);
    }

    /// Get element type name as string
    [[nodiscard]] std::string element_type_name() const {
        switch (element_type) {
            case VectorElementType::Float32:
                return "float32";
            case VectorElementType::Int8:
                return "int8";
            case VectorElementType::Bit:
                return "bit";
        }
        return "unknown";
    }

    /// Get distance metric name as string
    [[nodiscard]] std::string distance_metric_name() const {
        switch (distance_metric) {
            case DistanceMetric::L2:
                return "l2";
            case DistanceMetric::Cosine:
                return "cosine";
            case DistanceMetric::L1:
                return "l1";
        }
        return "unknown";
    }

    /// Validate the column definition
    [[nodiscard]] VoidResult validate() const {
        if (name.empty()) {
            return err_void(Error::invalid_argument("Column name cannot be empty"));
        }

        if (dimensions == 0) {
            return err_void(Error::invalid_argument("Vector dimensions must be > 0"));
        }

        // For bitvectors, dimensions must be divisible by 8
        if (element_type == VectorElementType::Bit && dimensions % CHAR_BIT != 0) {
            return err_void(Error::invalid_argument("Bitvector dimensions must be divisible by " +
                                                    std::to_string(CHAR_BIT)));
        }

        return ok();
    }
};

/// Partition column definition
struct PartitionColumnDefinition {
    std::string name;
    int type; // SQLite type (SQLITE_INTEGER, SQLITE_TEXT, etc.)

    PartitionColumnDefinition() = default; // members intentionally uninitialized

    PartitionColumnDefinition(std::string_view col_name, int sqlite_type)
        : name(col_name), type(sqlite_type) {}

    [[nodiscard]] std::string type_name() const {
        switch (type) {
            case SQLITE_INTEGER:
                return "INTEGER";
            case SQLITE_FLOAT:
                return "FLOAT";
            case SQLITE_TEXT:
                return "TEXT";
            case SQLITE_BLOB:
                return "BLOB";
            case SQLITE_NULL:
                return "NULL";
        }
        return "UNKNOWN";
    }
};

/// Auxiliary column definition (non-indexed columns)
struct AuxiliaryColumnDefinition {
    std::string name;
    int type; // SQLite type

    AuxiliaryColumnDefinition() = default; // members intentionally uninitialized

    AuxiliaryColumnDefinition(std::string_view col_name, int sqlite_type)
        : name(col_name), type(sqlite_type) {}

    [[nodiscard]] std::string type_name() const {
        switch (type) {
            case SQLITE_INTEGER:
                return "INTEGER";
            case SQLITE_FLOAT:
                return "FLOAT";
            case SQLITE_TEXT:
                return "TEXT";
            case SQLITE_BLOB:
                return "BLOB";
            case SQLITE_NULL:
                return "NULL";
        }
        return "UNKNOWN";
    }
};

/// Metadata column kind (for special system columns)
enum class MetadataColumnKind {
    Invalid = 0,
    Distance = 1,
    K = 2,
};

/// Metadata column definition
struct MetadataColumnDefinition {
    std::string name;
    MetadataColumnKind kind;

    MetadataColumnDefinition() = default; // members intentionally uninitialized

    MetadataColumnDefinition(std::string_view col_name, MetadataColumnKind col_kind)
        : name(col_name), kind(col_kind) {}

    [[nodiscard]] std::string kind_name() const {
        switch (kind) {
            case MetadataColumnKind::Distance:
                return "distance";
            case MetadataColumnKind::K:
                return "k";
            case MetadataColumnKind::Invalid:
                return "invalid";
        }
        return "unknown";
    }
};

// Helper functions

/// Calculate byte size for a vector based on element type and dimensions
[[nodiscard]] inline std::size_t vector_byte_size(VectorElementType element_type,
                                                  std::size_t dimensions) noexcept {
    switch (element_type) {
        case VectorElementType::Float32:
            return dimensions * sizeof(float);
        case VectorElementType::Int8:
            return dimensions * sizeof(std::int8_t);
        case VectorElementType::Bit:
            return dimensions / CHAR_BIT;
    }
    return 0;
}

/// Parse element type from string
[[nodiscard]] inline Result<VectorElementType> parse_element_type(std::string_view type_str) {
    if (type_str == "float32" || type_str == "float" || type_str == "f32") {
        return Result<VectorElementType>(VectorElementType::Float32);
    } else if (type_str == "int8" || type_str == "i8") {
        return Result<VectorElementType>(VectorElementType::Int8);
    } else if (type_str == "bit" || type_str == "binary") {
        return Result<VectorElementType>(VectorElementType::Bit);
    }
    return err<VectorElementType>(
        Error::invalid_argument("Unknown element type: " + std::string(type_str)));
}

/// Parse distance metric from string
[[nodiscard]] inline Result<DistanceMetric> parse_distance_metric(std::string_view metric_str) {
    if (metric_str == "l2" || metric_str == "euclidean") {
        return Result<DistanceMetric>(DistanceMetric::L2);
    } else if (metric_str == "cosine" || metric_str == "cos") {
        return Result<DistanceMetric>(DistanceMetric::Cosine);
    } else if (metric_str == "l1" || metric_str == "manhattan") {
        return Result<DistanceMetric>(DistanceMetric::L1);
    }
    return err<DistanceMetric>(
        Error::invalid_argument("Unknown distance metric: " + std::string(metric_str)));
}

} // namespace sqlite_vec_cpp::sqlite
