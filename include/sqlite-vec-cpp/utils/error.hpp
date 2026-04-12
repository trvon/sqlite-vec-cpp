#pragma once

#include <string>
#include <string_view>
#include <system_error>
#include <variant>

// C++23 std::expected or fallback
#ifdef HAS_CPP23_EXPECTED
#include <expected>
namespace sqlite_vec_cpp {
template <typename T, typename E> using Expected = std::expected<T, E>;

template <typename E> using Unexpected = std::unexpected<E>;
} // namespace sqlite_vec_cpp
#else
// Fallback: simplified expected implementation for now
// TODO: Use tl::expected or implement full expected<T,E>
namespace sqlite_vec_cpp {
template <typename E> class Unexpected {
public:
    explicit Unexpected(E error) : error_(std::move(error)) {}
    const E& error() const& { return error_; }
    E& error() & { return error_; }
    E&& error() && { return std::move(error_); }

private:
    E error_;
};

template <typename T, typename E> class Expected {
public:
    explicit Expected(T value) : storage_(std::in_place_index<0>, std::move(value)) {}

    explicit Expected(Unexpected<E> error)
        : storage_(std::in_place_index<1>, std::move(error.error())) {}

    explicit Expected(E error) : storage_(std::in_place_index<1>, std::move(error)) {}

    ~Expected() = default;

    Expected(const Expected&) = delete;
    Expected& operator=(const Expected&) = delete;
    Expected(Expected&&) noexcept = default;
    Expected& operator=(Expected&&) noexcept = default;

    bool has_value() const noexcept { return storage_.index() == 0; }
    explicit operator bool() const noexcept { return has_value(); }

    T& value() & {
        if (!has_value())
            throw std::runtime_error("Expected: no value");
        return std::get<0>(storage_);
    }

    const T& value() const& {
        if (!has_value())
            throw std::runtime_error("Expected: no value");
        return std::get<0>(storage_);
    }

    T&& value() && {
        if (!has_value())
            throw std::runtime_error("Expected: no value");
        return std::move(std::get<0>(storage_));
    }

    E& error() & {
        if (has_value())
            throw std::runtime_error("Expected: has value");
        return std::get<1>(storage_);
    }

    const E& error() const& {
        if (has_value())
            throw std::runtime_error("Expected: has value");
        return std::get<1>(storage_);
    }

    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }
    T&& operator*() && { return std::move(value()); }

    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

private:
    std::variant<T, E> storage_;
};

// Specialization for void
template <typename E> class Expected<void, E> {
public:
    Expected() : has_value_(true) {}

    explicit Expected(Unexpected<E> error) : has_value_(false), error_(std::move(error.error())) {}

    explicit Expected(E error) : has_value_(false), error_(std::move(error)) {}

    bool has_value() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    void value() const {
        if (!has_value_)
            throw std::runtime_error("Expected<void>: no value");
    }

    E& error() & {
        if (has_value_)
            throw std::runtime_error("Expected<void>: has value");
        return error_;
    }

    const E& error() const& {
        if (has_value_)
            throw std::runtime_error("Expected<void>: has value");
        return error_;
    }

private:
    bool has_value_;
    E error_;
};
} // namespace sqlite_vec_cpp
#endif

namespace sqlite_vec_cpp {

/// Error codes for sqlite-vec-cpp operations
enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    InvalidDimensions,
    DimensionMismatch,
    InvalidElementType,
    UnsupportedOperation,
    MemoryAllocation,
    SQLiteError,
    ExtensionNotLoaded,
    ParseError,
    InternalError,
};

/// Error information
struct Error {
    ErrorCode code{ErrorCode::InternalError};
    std::string message;
    int sqlite_code{0}; // Optional: store SQLite error code if applicable

    // Default constructor (needed for Expected)
    Error() = default;

    Error(ErrorCode c, std::string msg, int sql_code = 0)
        : code(c), message(std::move(msg)), sqlite_code(sql_code) {}

    [[nodiscard]] const char* what() const noexcept { return message.c_str(); }

    [[nodiscard]] static Error invalid_argument(std::string msg) {
        return Error{ErrorCode::InvalidArgument, std::move(msg)};
    }

    [[nodiscard]] static Error dimension_mismatch(std::size_t expected, std::size_t actual) {
        return Error{ErrorCode::DimensionMismatch, "Dimension mismatch: expected " +
                                                       std::to_string(expected) + ", got " +
                                                       std::to_string(actual)};
    }

    [[nodiscard]] static Error sqlite_error(std::string msg, int code) {
        return Error{ErrorCode::SQLiteError, std::move(msg), code};
    }
};

/// Result type for operations that may fail
template <typename T> using Result = Expected<T, Error>;

/// Void result (operation succeeded or error)
using VoidResult = Expected<void, Error>;

/// Helper to create success result
inline VoidResult ok() {
    return VoidResult{};
}

/// Helper to create error result
template <typename T> inline Result<T> err(Error error) {
    return Result<T>{Unexpected{std::move(error)}};
}

inline VoidResult err_void(Error error) {
    return VoidResult{Unexpected{std::move(error)}};
}

} // namespace sqlite_vec_cpp
