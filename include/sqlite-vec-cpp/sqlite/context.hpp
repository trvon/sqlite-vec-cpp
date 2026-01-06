#pragma once

#include <sqlite3.h>

#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include "../utils/error.hpp"
#include "value.hpp"

namespace sqlite_vec_cpp::sqlite {

/// RAII wrapper for sqlite3_context (function call context)
/// Note: sqlite3_context is managed by SQLite, this is a non-owning wrapper
class Context {
public:
    /// Construct from sqlite3_context pointer (non-owning)
    explicit Context(sqlite3_context* ctx) noexcept : ctx_(ctx) {}

    /// Deleted copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /// Move is okay
    Context(Context&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

    Context& operator=(Context&& other) noexcept {
        ctx_ = other.ctx_;
        other.ctx_ = nullptr;
        return *this;
    }

    /// Get raw SQLite context pointer
    [[nodiscard]] sqlite3_context* get() const noexcept { return ctx_; }
    [[nodiscard]] explicit operator bool() const noexcept { return ctx_ != nullptr; }

    // ========== Result Setters ==========

    /// Set result to NULL
    void result_null() const noexcept { sqlite3_result_null(ctx_); }

    /// Set integer result
    void result_int(int value) const noexcept { sqlite3_result_int(ctx_, value); }

    void result_int64(std::int64_t value) const noexcept { sqlite3_result_int64(ctx_, value); }

    /// Set floating-point result
    void result_double(double value) const noexcept { sqlite3_result_double(ctx_, value); }

    /// Set text result (makes a copy)
    void result_text(std::string_view text) const noexcept {
        sqlite3_result_text(ctx_, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
    }

    /// Set text result (takes ownership via SQLITE_TRANSIENT)
    void result_text(const std::string& text) const noexcept {
        sqlite3_result_text(ctx_, text.c_str(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
    }

    /// Set text result (no copy, pointer must remain valid)
    void result_text_static(const char* text) const noexcept {
        sqlite3_result_text(ctx_, text, -1, SQLITE_STATIC);
    }

    /// Set blob result (makes a copy)
    void result_blob(std::span<const std::uint8_t> blob) const noexcept {
        sqlite3_result_blob(ctx_, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
    }

    /// Set blob result with custom destructor
    template <typename Deleter>
    void result_blob(std::span<const std::uint8_t> blob,
                     [[maybe_unused]] Deleter&& deleter) const noexcept {
        // Note: For custom deleters, would need to wrap in a C callback
        // For now, just use SQLITE_TRANSIENT
        (void)deleter;
        result_blob(blob);
    }

    /// Set result with subtype (for vector extensions)
    void result_subtype(unsigned int subtype) const noexcept {
        sqlite3_result_subtype(ctx_, subtype);
    }

    /// Set blob result with subtype (for typed vectors)
    void result_blob_with_subtype(std::span<const std::uint8_t> blob,
                                  unsigned int subtype) const noexcept {
        sqlite3_result_blob(ctx_, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
        sqlite3_result_subtype(ctx_, subtype);
    }

    /// Set error result
    void result_error(std::string_view msg) const noexcept {
        sqlite3_result_error(ctx_, msg.data(), static_cast<int>(msg.size()));
    }

    void result_error_code(int error_code) const noexcept {
        sqlite3_result_error_code(ctx_, error_code);
    }

    void result_error_nomem() const noexcept { sqlite3_result_error_nomem(ctx_); }

    void result_error_toobig() const noexcept { sqlite3_result_error_toobig(ctx_); }

    /// Set error from Error type
    void result_error(const Error& error) const noexcept {
        result_error(error.message);
        if (error.sqlite_code != 0) {
            result_error_code(error.sqlite_code);
        }
    }

    // ========== Context Queries ==========

    /// Get user data
    [[nodiscard]] void* user_data() const noexcept { return sqlite3_user_data(ctx_); }

    template <typename T> [[nodiscard]] T* user_data() const noexcept {
        return static_cast<T*>(sqlite3_user_data(ctx_));
    }

    /// Get database connection
    [[nodiscard]] sqlite3* db_handle() const noexcept { return sqlite3_context_db_handle(ctx_); }

    /// Get aggregate context
    [[nodiscard]] void* aggregate_context(int n_bytes) const noexcept {
        return sqlite3_aggregate_context(ctx_, n_bytes);
    }

    template <typename T> [[nodiscard]] T* aggregate_context() const noexcept {
        return static_cast<T*>(sqlite3_aggregate_context(ctx_, sizeof(T)));
    }

    // ========== Auxiliary Data ==========

    /// Get auxiliary data
    [[nodiscard]] void* get_auxdata(int n) const noexcept { return sqlite3_get_auxdata(ctx_, n); }

    template <typename T> [[nodiscard]] T* get_auxdata(int n) const noexcept {
        return static_cast<T*>(sqlite3_get_auxdata(ctx_, n));
    }

    /// Set auxiliary data with destructor
    void set_auxdata(int n, void* data, void (*destructor)(void*)) const noexcept {
        sqlite3_set_auxdata(ctx_, n, data, destructor);
    }

    // ========== Convenience Wrappers ==========

    /// Set result from Result<T> monad
    template <typename T> void result_from(const Result<T>& result) const noexcept {
        if (result) {
            if constexpr (std::is_same_v<T, int>) {
                result_int(result.value());
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                result_int64(result.value());
            } else if constexpr (std::is_same_v<T, double>) {
                result_double(result.value());
            } else if constexpr (std::is_same_v<T, std::string>) {
                result_text(result.value());
            } else {
                // Generic: try to convert to string or handle specially
                result_null();
            }
        } else {
            result_error(result.error());
        }
    }

    /// Set result from VoidResult
    void result_from(const VoidResult& result) const noexcept {
        if (!result) {
            result_error(result.error());
        }
        // For void success, do nothing (caller sets result)
    }

private:
    sqlite3_context* ctx_;
};

} // namespace sqlite_vec_cpp::sqlite
