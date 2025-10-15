#pragma once

#include <sqlite3.h>

#include "../utils/error.hpp"
#include "enhanced_functions.hpp"
#include "functions.hpp"
#include "utility_functions.hpp"
#include "vec0_module.hpp"

namespace sqlite_vec_cpp::sqlite {

/// Register all vector distance functions and vec0 virtual table module with SQLite
inline Result<void> register_all_functions(sqlite3* db) {
    if (!db) {
        return err<void>(Error::invalid_argument("database handle is null"));
    }

    // Register vec0 virtual table module first (critical for CREATE VIRTUAL TABLE)
    auto vec0_result = register_vec0_module(db);
    if (!vec0_result) {
        return vec0_result;
    }

    // Register core distance functions (compatible with original sqlite-vec)
    int rc =
        sqlite3_create_function_v2(db, "vec_distance_l2", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                   nullptr, vec_distance_l2, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_distance_l2", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_distance_l1", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                    nullptr, vec_distance_l1, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_distance_l1", rc));
    }

    rc =
        sqlite3_create_function_v2(db, "vec_distance_cosine", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                   nullptr, vec_distance_cosine, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_distance_cosine", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_distance_hamming", 2,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr,
                                    vec_distance_hamming, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_distance_hamming", rc));
    }

    // Register utility functions
    rc = sqlite3_create_function_v2(db, "vec_length", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                    nullptr, vec_length, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_length", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_type", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr,
                                    vec_type, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_type", rc));
    }

    // Register enhanced functions (C++20/23 specific features)
    rc = sqlite3_create_function_v2(db, "vec_dot", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr,
                                    vec_dot, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_dot", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_magnitude", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                    nullptr, vec_magnitude, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_magnitude", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_scale", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr,
                                    vec_scale, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_scale", rc));
    }

    rc = sqlite3_create_function_v2(db, "vec_mean", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr,
                                    vec_mean, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_mean", rc));
    }

    return Result<void>();
}

} // namespace sqlite_vec_cpp::sqlite
