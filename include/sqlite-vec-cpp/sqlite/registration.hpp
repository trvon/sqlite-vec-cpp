#pragma once

#include <sqlite3.h>

// Compatibility defines for older SQLite versions
// SQLITE_SUBTYPE was added in SQLite 3.30.0
#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

// SQLITE_RESULT_SUBTYPE was added in SQLite 3.45.0
#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

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

    // Register vec_f32 function (vector creation from JSON)
    rc = sqlite3_create_function_v2(db, "vec_f32", 1,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE |
                                        SQLITE_RESULT_SUBTYPE,
                                    nullptr, vec_f32, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_f32", rc));
    }

    // Register vec_f32_simple function (for vec0 compatibility)
    // This version doesn't set subtype so virtual table xUpdate can access blob
    rc = sqlite3_create_function_v2(db, "vec_f32_simple", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                    nullptr, vec_f32_simple, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_f32_simple", rc));
    }

    // Register vec_int8 function (int8 vector creation from JSON)
    rc = sqlite3_create_function_v2(db, "vec_int8", 1,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE |
                                        SQLITE_RESULT_SUBTYPE,
                                    nullptr, vec_int8, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_int8", rc));
    }

    // Register vec_bit function (bit vector creation from blob)
    rc = sqlite3_create_function_v2(db, "vec_bit", 1,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE |
                                        SQLITE_RESULT_SUBTYPE,
                                    nullptr, vec_bit, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_bit", rc));
    }

    // Register vec_f32_simple function (for vec0 compatibility)
    // This version doesn't set subtype so virtual table xUpdate can access blob
    rc = sqlite3_create_function_v2(db, "vec_f32_simple", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                    nullptr, vec_f32_simple, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_f32_simple", rc));
    }

    // Register vec_int8 function (int8 vector creation from JSON)
    rc = sqlite3_create_function_v2(db, "vec_int8", 1,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE |
                                        SQLITE_RESULT_SUBTYPE,
                                    nullptr, vec_int8, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_int8", rc));
    }

    // Register vec_bit function (bit vector creation from blob)
    rc = sqlite3_create_function_v2(db, "vec_bit", 1,
                                    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE |
                                        SQLITE_RESULT_SUBTYPE,
                                    nullptr, vec_bit, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec_bit", rc));
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
