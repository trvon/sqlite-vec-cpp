#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include "../utils/error.hpp"

namespace sqlite_vec_cpp::sqlite {

/// RAII wrapper for sqlite3_value (read-only)
/// Note: sqlite3_value objects are managed by SQLite, not by us
/// This is a non-owning wrapper that provides type-safe access
class Value {
public:
    /// Construct from sqlite3_value pointer (non-owning)
    explicit Value(sqlite3_value* value) noexcept : value_(value) {}

    /// Deleted copy (to prevent accidental copies of non-owning wrapper)
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    /// Move is okay (just transfers the pointer)
    Value(Value&& other) noexcept : value_(other.value_) { other.value_ = nullptr; }

    Value& operator=(Value&& other) noexcept {
        value_ = other.value_;
        other.value_ = nullptr;
        return *this;
    }

    /// Get raw SQLite value pointer
    [[nodiscard]] sqlite3_value* get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }

    /// Type queries
    [[nodiscard]] int type() const noexcept { return sqlite3_value_type(value_); }

    [[nodiscard]] int subtype() const noexcept { return sqlite3_value_subtype(value_); }

    [[nodiscard]] bool is_null() const noexcept { return type() == SQLITE_NULL; }

    [[nodiscard]] bool is_integer() const noexcept { return type() == SQLITE_INTEGER; }

    [[nodiscard]] bool is_float() const noexcept { return type() == SQLITE_FLOAT; }

    [[nodiscard]] bool is_text() const noexcept { return type() == SQLITE_TEXT; }

    [[nodiscard]] bool is_blob() const noexcept { return type() == SQLITE_BLOB; }

    /// Value extractors
    [[nodiscard]] std::int64_t as_int64() const noexcept { return sqlite3_value_int64(value_); }

    [[nodiscard]] int as_int() const noexcept { return sqlite3_value_int(value_); }

    [[nodiscard]] double as_double() const noexcept { return sqlite3_value_double(value_); }

    [[nodiscard]] std::string_view as_text() const noexcept {
        const unsigned char* text = sqlite3_value_text(value_);
        if (!text)
            return {};
        int bytes = sqlite3_value_bytes(value_);
        return {reinterpret_cast<const char*>(text), static_cast<std::size_t>(bytes)};
    }

    [[nodiscard]] std::span<const std::uint8_t> as_blob() const noexcept {
        const void* blob = sqlite3_value_blob(value_);
        if (!blob)
            return {};
        int bytes = sqlite3_value_bytes(value_);
        return {static_cast<const std::uint8_t*>(blob), static_cast<std::size_t>(bytes)};
    }

    /// Size queries
    [[nodiscard]] int bytes() const noexcept { return sqlite3_value_bytes(value_); }

    /// Numeric coercion (forces type conversion)
    [[nodiscard]] int numeric_type() const noexcept { return sqlite3_value_numeric_type(value_); }

    /// No-change flag for optimizations
    [[nodiscard]] bool no_change() const noexcept { return sqlite3_value_nochange(value_) != 0; }

private:
    sqlite3_value* value_;
};

/// Helper for working with multiple values
class ValueArray {
public:
    ValueArray(sqlite3_value** values, int count) noexcept : values_(values), count_(count) {}

    [[nodiscard]] int size() const noexcept { return count_; }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    [[nodiscard]] Value operator[](int index) const noexcept { return Value(values_[index]); }

    [[nodiscard]] Value at(int index) const {
        if (index < 0 || index >= count_) {
            throw std::out_of_range("ValueArray index out of range");
        }
        return Value(values_[index]);
    }

    /// Iterator support
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Value;
        using difference_type = std::ptrdiff_t;
        using pointer = void;    // Can't return pointer to temporary
        using reference = Value; // Returns Value by value

        explicit Iterator(sqlite3_value** ptr) : ptr_(ptr) {}

        reference operator*() const { return Value(*ptr_); }
        Iterator& operator++() {
            ++ptr_;
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++ptr_;
            return tmp;
        }
        Iterator& operator--() {
            --ptr_;
            return *this;
        }
        Iterator operator--(int) {
            Iterator tmp = *this;
            --ptr_;
            return tmp;
        }
        Iterator& operator+=(difference_type n) {
            ptr_ += n;
            return *this;
        }
        Iterator& operator-=(difference_type n) {
            ptr_ -= n;
            return *this;
        }
        friend Iterator operator+(Iterator it, difference_type n) { return Iterator(it.ptr_ + n); }
        friend Iterator operator+(difference_type n, Iterator it) { return Iterator(it.ptr_ + n); }
        friend Iterator operator-(Iterator it, difference_type n) { return Iterator(it.ptr_ - n); }
        friend difference_type operator-(Iterator a, Iterator b) { return a.ptr_ - b.ptr_; }
        reference operator[](difference_type n) const { return Value(ptr_[n]); }

        bool operator==(const Iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const Iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const Iterator& other) const { return ptr_ < other.ptr_; }
        bool operator<=(const Iterator& other) const { return ptr_ <= other.ptr_; }
        bool operator>(const Iterator& other) const { return ptr_ > other.ptr_; }
        bool operator>=(const Iterator& other) const { return ptr_ >= other.ptr_; }

    private:
        sqlite3_value** ptr_;
    };

    [[nodiscard]] Iterator begin() const noexcept { return Iterator(values_); }
    [[nodiscard]] Iterator end() const noexcept { return Iterator(values_ + count_); }

private:
    sqlite3_value** values_;
    int count_;
};

} // namespace sqlite_vec_cpp::sqlite
