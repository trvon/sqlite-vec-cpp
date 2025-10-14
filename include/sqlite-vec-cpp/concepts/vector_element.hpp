#pragma once

#include <cstdint>
#include <type_traits>

namespace sqlite_vec_cpp::concepts {

/// Concept for types that can be used as vector elements
/// Matches: float, int8_t, int16_t, uint8_t, etc.
template <typename T>
concept VectorElement = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

/// Concept for floating-point vector elements
template <typename T>
concept FloatingPointElement = VectorElement<T> && std::is_floating_point_v<T>;

/// Concept for integer vector elements
template <typename T>
concept IntegerElement = VectorElement<T> && std::is_integral_v<T> && !std::is_same_v<T, bool>;

/// Concept for signed integer vector elements
template <typename T>
concept SignedIntegerElement = IntegerElement<T> && std::is_signed_v<T>;

/// Concept for unsigned integer vector elements
template <typename T>
concept UnsignedIntegerElement = IntegerElement<T> && std::is_unsigned_v<T>;

// Type traits for element properties
namespace traits {

/// Get the storage size in bytes for a vector element type
template <VectorElement T> constexpr std::size_t element_size_v = sizeof(T);

/// Check if type is a supported float type (float or double)
template <typename T> constexpr bool is_float_v = std::is_same_v<T, float>;

/// Check if type is int8_t
template <typename T> constexpr bool is_int8_v = std::is_same_v<T, std::int8_t>;

/// Check if type is int16_t
template <typename T> constexpr bool is_int16_v = std::is_same_v<T, std::int16_t>;

/// Check if type is uint8_t
template <typename T> constexpr bool is_uint8_v = std::is_same_v<T, std::uint8_t>;

/// Check if type requires SIMD alignment
template <VectorElement T> constexpr bool requires_simd_alignment_v = is_float_v<T> || is_int8_v<T>;

/// Get optimal SIMD vector width for type (elements per SIMD register)
template <VectorElement T>
constexpr std::size_t simd_width_v = []() {
    if constexpr (std::is_same_v<T, float>) {
        return 8; // AVX: 256-bit / 32-bit = 8 floats
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return 16; // NEON: 128-bit / 8-bit = 16 int8s
    } else {
        return 1; // No SIMD for other types (fallback)
    }
}();

} // namespace traits

// Static assertions to validate concept requirements
static_assert(VectorElement<float>, "float must satisfy VectorElement");
static_assert(VectorElement<std::int8_t>, "int8_t must satisfy VectorElement");
static_assert(VectorElement<std::int16_t>, "int16_t must satisfy VectorElement");
static_assert(VectorElement<std::uint8_t>, "uint8_t must satisfy VectorElement");
static_assert(!VectorElement<bool>, "bool must NOT satisfy VectorElement");
static_assert(!VectorElement<void*>, "void* must NOT satisfy VectorElement");

static_assert(FloatingPointElement<float>, "float must satisfy FloatingPointElement");
static_assert(!FloatingPointElement<std::int8_t>, "int8_t must NOT satisfy FloatingPointElement");

static_assert(IntegerElement<std::int8_t>, "int8_t must satisfy IntegerElement");
static_assert(!IntegerElement<float>, "float must NOT satisfy IntegerElement");

} // namespace sqlite_vec_cpp::concepts
