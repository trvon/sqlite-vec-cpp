#pragma once

#include <sqlite3.h>
#include <cstdarg>
#include <memory>
#include <string>
#include <string_view>
#include "../utils/error.hpp"

namespace sqlite_vec_cpp::sqlite {

/// RAII wrapper for sqlite3_vtab (virtual table base)
/// Note: This is managed by SQLite but we provide a type-safe wrapper
class VTab {
public:
    /// Construct from sqlite3_vtab pointer (non-owning)
    explicit VTab(sqlite3_vtab* vtab) noexcept : vtab_(vtab) {}

    /// Deleted copy
    VTab(const VTab&) = delete;
    VTab& operator=(const VTab&) = delete;

    /// Move is okay
    VTab(VTab&& other) noexcept : vtab_(other.vtab_) { other.vtab_ = nullptr; }

    VTab& operator=(VTab&& other) noexcept {
        vtab_ = other.vtab_;
        other.vtab_ = nullptr;
        return *this;
    }

    /// Get raw SQLite vtab pointer
    [[nodiscard]] sqlite3_vtab* get() const noexcept { return vtab_; }
    [[nodiscard]] explicit operator bool() const noexcept { return vtab_ != nullptr; }

    /// Set error message (managed by SQLite)
    void set_error(std::string_view msg) const noexcept {
        sqlite3_free(vtab_->zErrMsg);
        vtab_->zErrMsg = sqlite3_mprintf("%.*s", static_cast<int>(msg.size()), msg.data());
    }

    void set_error(const char* format, ...) const noexcept {
        va_list args;
        va_start(args, format);
        sqlite3_free(vtab_->zErrMsg);
        vtab_->zErrMsg = sqlite3_vmprintf(format, args);
        va_end(args);
    }

    /// Set error from Error type
    void set_error(const Error& error) const noexcept { set_error(error.message); }

    /// Clear error
    void clear_error() const noexcept {
        sqlite3_free(vtab_->zErrMsg);
        vtab_->zErrMsg = nullptr;
    }

    /// Get current error message
    [[nodiscard]] const char* error_msg() const noexcept { return vtab_->zErrMsg; }

    [[nodiscard]] bool has_error() const noexcept { return vtab_->zErrMsg != nullptr; }

private:
    sqlite3_vtab* vtab_;
};

/// RAII wrapper for sqlite3_vtab_cursor (virtual table cursor)
class VTabCursor {
public:
    explicit VTabCursor(sqlite3_vtab_cursor* cursor) noexcept : cursor_(cursor) {}

    VTabCursor(const VTabCursor&) = delete;
    VTabCursor& operator=(const VTabCursor&) = delete;

    VTabCursor(VTabCursor&& other) noexcept : cursor_(other.cursor_) { other.cursor_ = nullptr; }

    VTabCursor& operator=(VTabCursor&& other) noexcept {
        cursor_ = other.cursor_;
        other.cursor_ = nullptr;
        return *this;
    }

    [[nodiscard]] sqlite3_vtab_cursor* get() const noexcept { return cursor_; }
    [[nodiscard]] explicit operator bool() const noexcept { return cursor_ != nullptr; }

    /// Get parent virtual table
    [[nodiscard]] sqlite3_vtab* vtab() const noexcept { return cursor_->pVtab; }

private:
    sqlite3_vtab_cursor* cursor_;
};

/// Helper for virtual table index info
/// Note: Some detailed constraint types may not be available in all SQLite versions
class IndexInfo {
public:
    explicit IndexInfo(sqlite3_index_info* info) noexcept : info_(info) {}

    [[nodiscard]] sqlite3_index_info* get() const noexcept { return info_; }

    // Basic constraint access
    [[nodiscard]] int num_constraints() const noexcept { return info_->nConstraint; }

    // Order by access
    [[nodiscard]] int num_order_by() const noexcept { return info_->nOrderBy; }

    // Index output
    void set_index_num(int num) const noexcept { info_->idxNum = num; }

    void set_index_str(const char* str) const noexcept {
        info_->idxStr = sqlite3_mprintf("%s", str);
        info_->needToFreeIdxStr = 1;
    }

    void set_order_by_consumed(bool consumed) const noexcept {
        info_->orderByConsumed = consumed ? 1 : 0;
    }

    void set_estimated_cost(double cost) const noexcept { info_->estimatedCost = cost; }

    void set_estimated_rows(std::int64_t rows) const noexcept { info_->estimatedRows = rows; }

private:
    sqlite3_index_info* info_;
};

} // namespace sqlite_vec_cpp::sqlite
