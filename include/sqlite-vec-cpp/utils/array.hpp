#pragma once

#include <cstring>
#include <memory>
#include <span>
#include <vector>
#include "../concepts/vector_element.hpp"
#include "error.hpp"

namespace sqlite_vec_cpp::utils {

/// Modern C++ replacement for struct Array from sqlite-vec.c
/// Uses std::vector internally with SQLite-compatible interface
template <typename T> class Array {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    /// Construct with initial capacity (mirrors array_init)
    explicit Array(size_type init_capacity = 0) {
        if (init_capacity > 0) {
            data_.reserve(init_capacity);
        }
    }

    /// Append element (mirrors array_append)
    /// Returns VoidResult for error handling
    VoidResult append(const T& element) {
        try {
            data_.push_back(element);
            return ok();
        } catch (const std::bad_alloc&) {
            return err_void(Error{ErrorCode::MemoryAllocation, "Failed to append to array"});
        }
    }

    /// Append element with move semantics
    VoidResult append(T&& element) {
        try {
            data_.push_back(std::move(element));
            return ok();
        } catch (const std::bad_alloc&) {
            return err_void(Error{ErrorCode::MemoryAllocation, "Failed to append to array"});
        }
    }

    /// Append raw memory (for C compatibility)
    VoidResult append_bytes(const void* element_bytes, size_type element_size) {
        if (element_size != sizeof(T)) {
            return err_void(Error{ErrorCode::InvalidArgument, "Element size mismatch"});
        }
        try {
            T element;
            std::memcpy(&element, element_bytes, sizeof(T));
            data_.push_back(element);
            return ok();
        } catch (const std::bad_alloc&) {
            return err_void(Error{ErrorCode::MemoryAllocation, "Failed to append to array"});
        }
    }

    /// Access elements
    [[nodiscard]] reference operator[](size_type idx) { return data_[idx]; }
    [[nodiscard]] const_reference operator[](size_type idx) const { return data_[idx]; }

    [[nodiscard]] reference at(size_type idx) { return data_.at(idx); }
    [[nodiscard]] const_reference at(size_type idx) const { return data_.at(idx); }

    /// Size queries
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] size_type length() const noexcept { return data_.size(); }
    [[nodiscard]] size_type capacity() const noexcept { return data_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    /// Reserve capacity
    void reserve(size_type new_capacity) { data_.reserve(new_capacity); }

    /// Clear contents
    void clear() noexcept { data_.clear(); }

    /// Raw data access (for C API compatibility)
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    /// Get as span
    [[nodiscard]] std::span<T> span() noexcept { return {data_.data(), data_.size()}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {data_.data(), data_.size()}; }

    /// Iterators
    [[nodiscard]] iterator begin() noexcept { return data_.begin(); }
    [[nodiscard]] iterator end() noexcept { return data_.end(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return data_.end(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return data_.cend(); }

    /// Get underlying vector (for advanced usage)
    [[nodiscard]] std::vector<T>& vector() noexcept { return data_; }
    [[nodiscard]] const std::vector<T>& vector() const noexcept { return data_; }

private:
    std::vector<T> data_;
};

/// Type-erased array for runtime polymorphism (if needed)
/// Similar to original void* based Array
class DynamicArray {
public:
    explicit DynamicArray(std::size_t element_size, std::size_t init_capacity = 0)
        : element_size_(element_size), length_(0) {
        if (init_capacity > 0) {
            data_.reserve(init_capacity * element_size);
        }
    }

    VoidResult append(const void* element) {
        try {
            const auto* bytes = static_cast<const std::uint8_t*>(element);
            data_.insert(data_.end(), bytes, bytes + element_size_);
            ++length_;
            return ok();
        } catch (const std::bad_alloc&) {
            return err_void(
                Error{ErrorCode::MemoryAllocation, "Failed to append to dynamic array"});
        }
    }

    [[nodiscard]] void* get(std::size_t index) {
        if (index >= length_) {
            return nullptr;
        }
        return &data_[index * element_size_];
    }

    [[nodiscard]] const void* get(std::size_t index) const {
        if (index >= length_) {
            return nullptr;
        }
        return &data_[index * element_size_];
    }

    [[nodiscard]] std::size_t size() const noexcept { return length_; }
    [[nodiscard]] std::size_t length() const noexcept { return length_; }
    [[nodiscard]] std::size_t element_size() const noexcept { return element_size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return data_.capacity() / element_size_; }
    [[nodiscard]] bool empty() const noexcept { return length_ == 0; }

    void clear() noexcept {
        data_.clear();
        length_ = 0;
    }

    [[nodiscard]] void* data() noexcept { return data_.data(); }
    [[nodiscard]] const void* data() const noexcept { return data_.data(); }

private:
    std::size_t element_size_;
    std::size_t length_;
    std::vector<std::uint8_t> data_;
};

} // namespace sqlite_vec_cpp::utils
