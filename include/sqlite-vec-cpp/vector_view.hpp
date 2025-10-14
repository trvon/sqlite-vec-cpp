#pragma once

#include <cassert>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>
#include "concepts/vector_element.hpp"

namespace sqlite_vec_cpp {

/// Type-safe view over vector data
/// Wraps std::span with additional vector-specific operations
template <concepts::VectorElement T> class VectorView {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using span_type = std::span<T>;
    using const_span_type = std::span<const T>;

    /// Construct from span
    explicit constexpr VectorView(std::span<T> data) noexcept : data_(data) {}

    /// Construct from pointer and size
    constexpr VectorView(T* data, size_type size) noexcept : data_(data, size) {}

    /// Construct from std::vector
    template <typename Alloc>
    explicit VectorView(std::vector<value_type, Alloc>& vec) noexcept
        : data_(vec.data(), vec.size()) {}

    /// Construct from const std::vector (for const T)
    template <typename Alloc>
    requires std::is_const_v<T>
    explicit VectorView(const std::vector<value_type, Alloc>& vec) noexcept
        : data_(vec.data(), vec.size()) {}

    // Accessors
    [[nodiscard]] constexpr auto data() const noexcept { return data_.data(); }
    [[nodiscard]] constexpr size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] constexpr size_type dimensions() const noexcept { return data_.size(); }
    [[nodiscard]] constexpr bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] constexpr span_type span() noexcept { return data_; }
    [[nodiscard]] constexpr const_span_type span() const noexcept { return data_; }

    // Element access
    [[nodiscard]] constexpr T& operator[](size_type idx) noexcept {
        assert(idx < size());
        return data_[idx];
    }

    [[nodiscard]] constexpr const T& operator[](size_type idx) const noexcept {
        assert(idx < size());
        return data_[idx];
    }

    [[nodiscard]] constexpr T& at(size_type idx) {
        if (idx >= size()) {
            throw std::out_of_range("VectorView::at: index out of range");
        }
        return data_[idx];
    }

    [[nodiscard]] constexpr const T& at(size_type idx) const {
        if (idx >= size()) {
            throw std::out_of_range("VectorView::at: index out of range");
        }
        return data_[idx];
    }

    // Iterators
    [[nodiscard]] constexpr auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] constexpr auto end() noexcept { return data_.end(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return data_.end(); }
    [[nodiscard]] constexpr auto cbegin() const noexcept { return data_.begin(); }
    [[nodiscard]] constexpr auto cend() const noexcept { return data_.end(); }

    // Conversion to byte representation (for SQLite storage)
    [[nodiscard]] std::vector<std::uint8_t> to_blob() const {
        std::vector<std::uint8_t> blob(size() * sizeof(T));
        std::memcpy(blob.data(), data_.data(), blob.size());
        return blob;
    }

    [[nodiscard]] std::span<const std::uint8_t> as_bytes() const noexcept {
        return std::as_bytes(data_);
    }

    /// Validate dimensions match expected
    [[nodiscard]] constexpr bool has_dimensions(size_type expected) const noexcept {
        return size() == expected;
    }

    /// Check if SIMD-aligned
    [[nodiscard]] bool is_simd_aligned() const noexcept {
        constexpr size_type alignment = concepts::traits::simd_width_v<T>;
        return size() % alignment == 0 &&
               reinterpret_cast<std::uintptr_t>(data()) % (alignment * sizeof(T)) == 0;
    }

private:
    std::span<T> data_;
};

// Deduction guides
template <typename T> VectorView(T*, std::size_t) -> VectorView<T>;

template <typename T, typename Alloc> VectorView(std::vector<T, Alloc>&) -> VectorView<T>;

template <typename T, typename Alloc>
VectorView(const std::vector<T, Alloc>&) -> VectorView<const T>;

// Helper functions

/// Create VectorView from raw SQLite blob
template <concepts::VectorElement T>
[[nodiscard]] VectorView<const T> vector_view_from_blob(const void* blob, std::size_t byte_size) {
    assert(byte_size % sizeof(T) == 0 && "Blob size must be multiple of element size");
    const auto* typed_data = static_cast<const T*>(blob);
    const std::size_t num_elements = byte_size / sizeof(T);
    return VectorView<const T>{typed_data, num_elements};
}

/// Convert blob to vector (copies data)
template <concepts::VectorElement T>
[[nodiscard]] std::vector<T> vector_from_blob(const void* blob, std::size_t byte_size) {
    assert(byte_size % sizeof(T) == 0 && "Blob size must be multiple of element size");
    const std::size_t num_elements = byte_size / sizeof(T);
    std::vector<T> result(num_elements);
    std::memcpy(result.data(), blob, byte_size);
    return result;
}

/// Validate two vectors have same dimensions
template <concepts::VectorElement T, concepts::VectorElement U>
[[nodiscard]] constexpr bool same_dimensions(const VectorView<T>& a,
                                             const VectorView<U>& b) noexcept {
    return a.size() == b.size();
}

} // namespace sqlite_vec_cpp
