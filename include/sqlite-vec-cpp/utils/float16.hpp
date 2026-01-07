#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace sqlite_vec_cpp::utils {

/// IEEE 754 half-precision floating point (16-bit)
struct float16_t {
    uint16_t bits = 0;

    float16_t() = default;

    explicit float16_t(uint16_t b) : bits(b) {}

    static float16_t from_float(float f) noexcept {
        uint32_t x = std::bit_cast<uint32_t>(f);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = x & 0x7FFFFF;

        uint16_t result;
        if (exp <= 0) {
            if (exp < -10) {
                result = static_cast<uint16_t>(sign);
            } else {
                mantissa = (mantissa | 0x800000) >> (1 - exp);
                result = static_cast<uint16_t>(sign | (mantissa >> 13));
            }
        } else if (exp >= 31) {
            result = static_cast<uint16_t>(sign | 0x7C00);
        } else {
            result = static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
        }
        return float16_t(result);
    }

    [[nodiscard]] float to_float() const noexcept {
        uint32_t sign = (bits & 0x8000) << 16;
        uint32_t exp = (bits >> 10) & 0x1F;
        uint32_t mantissa = bits & 0x3FF;

        uint32_t result;
        if (exp == 0) {
            if (mantissa == 0) {
                result = sign;
            } else {
                exp = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3FF;
                result = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exp == 31) {
            result = sign | 0x7F800000 | (mantissa << 13);
        } else {
            result = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }

        return std::bit_cast<float>(result);
    }

    /// Explicit conversion to float for use in static_cast<float>(float16_t)
    explicit operator float() const noexcept { return to_float(); }

    auto operator<=>(const float16_t& other) const noexcept = default;

    bool operator==(const float16_t& other) const noexcept = default;
};

static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");

inline std::vector<float16_t> to_float16(std::span<const float> src) {
    std::vector<float16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = float16_t::from_float(src[i]);
    }
    return dst;
}

inline std::vector<float> to_float32(std::span<const float16_t> src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i].to_float();
    }
    return dst;
}

inline void convert_to_float16_inplace(std::span<float16_t> dst, std::span<const float> src) {
    assert(dst.size() == src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = float16_t::from_float(src[i]);
    }
}

inline void convert_to_float32_inplace(std::span<float> dst, std::span<const float16_t> src) {
    assert(dst.size() == src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i].to_float();
    }
}

} // namespace sqlite_vec_cpp::utils
