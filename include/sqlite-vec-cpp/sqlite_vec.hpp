#pragma once

// Main public API header for sqlite-vec-cpp

// Core concepts
#include "concepts/distance_metric.hpp"
#include "concepts/vector_element.hpp"

// Vector abstractions
#include "vector_view.hpp"

// Utilities
#include "utils/array.hpp"
#include "utils/error.hpp"

// Distance metrics
#include "distances/cosine.hpp"
#include "distances/hamming.hpp"
#include "distances/l1.hpp"
#include "distances/l2.hpp"

// SQLite integration
#include "sqlite/column_defs.hpp"
#include "sqlite/context.hpp"
#include "sqlite/enhanced_functions.hpp"
#include "sqlite/functions.hpp"
#include "sqlite/parsers.hpp"
#include "sqlite/registration.hpp"
#include "sqlite/utility_functions.hpp"
#include "sqlite/value.hpp"
#include "sqlite/vtab.hpp"

// C API compatibility (to be implemented)
extern "C" {
// Forward declaration - will be defined in src/sqlite_vec_c_api.cpp
struct sqlite3;
struct sqlite3_api_routines;

// Main entry point (maintains compatibility with original sqlite-vec)
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_vec_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* pApi);

// Additional C API functions
int sqlite3_vec_distance_l2(const void* vec1, size_t size1, const void* vec2, size_t size2,
                            float* result);
int sqlite3_vec_distance_cosine(const void* vec1, size_t size1, const void* vec2, size_t size2,
                                float* result);
}

namespace sqlite_vec_cpp {

// Version information
constexpr const char* VERSION = "0.1.0-cpp";
constexpr const char* SOURCE = "sqlite-vec-cpp";

// Version components
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

} // namespace sqlite_vec_cpp
