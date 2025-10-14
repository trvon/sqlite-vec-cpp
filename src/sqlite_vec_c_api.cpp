// SQLite-Vec C API Compatibility Layer
// Maintains backward compatibility with original sqlite-vec C API
// while leveraging modern C++20/23 implementation internally

#include <sqlite3ext.h>
#include <sqlite-vec-cpp/sqlite/registration.hpp>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

SQLITE_EXTENSION_INIT1

// External C API entry point (maintains compatibility)
extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_vec_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi
) {
    SQLITE_EXTENSION_INIT2(pApi);

    try {
        // Register all vector functions from C++ implementation
        auto result = sqlite_vec_cpp::sqlite::register_all_functions(db);

        if (!result) {
            if (pzErrMsg) {
                *pzErrMsg = sqlite3_mprintf("Failed to initialize sqlite-vec: %s",
                                            result.error().message.c_str());
            }
            return SQLITE_ERROR;
        }

        return SQLITE_OK;
    } catch (const std::exception& e) {
        if (pzErrMsg) {
            *pzErrMsg = sqlite3_mprintf("Exception during sqlite-vec initialization: %s", e.what());
        }
        return SQLITE_ERROR;
    } catch (...) {
        if (pzErrMsg) {
            *pzErrMsg = sqlite3_mprintf("Unknown exception during sqlite-vec initialization");
        }
        return SQLITE_ERROR;
    }
}

// Additional C API functions for YAMS compatibility
int sqlite3_vec_distance_l2(const void* vec1, size_t size1, const void* vec2, size_t size2,
                            float* result) {
    try {
        if (!vec1 || !vec2 || !result) {
            return SQLITE_ERROR;
        }

        size_t dim1 = size1 / sizeof(float);
        size_t dim2 = size2 / sizeof(float);

        if (dim1 != dim2) {
            return SQLITE_ERROR;
        }

        std::span<const float> v1(static_cast<const float*>(vec1), dim1);
        std::span<const float> v2(static_cast<const float*>(vec2), dim2);

        *result = sqlite_vec_cpp::distances::l2_distance(v1, v2);
        return SQLITE_OK;
    } catch (...) {
        return SQLITE_ERROR;
    }
}

int sqlite3_vec_distance_cosine(const void* vec1, size_t size1, const void* vec2, size_t size2,
                                float* result) {
    try {
        if (!vec1 || !vec2 || !result) {
            return SQLITE_ERROR;
        }

        size_t dim1 = size1 / sizeof(float);
        size_t dim2 = size2 / sizeof(float);

        if (dim1 != dim2) {
            return SQLITE_ERROR;
        }

        std::span<const float> v1(static_cast<const float*>(vec1), dim1);
        std::span<const float> v2(static_cast<const float*>(vec2), dim2);

        *result = sqlite_vec_cpp::distances::cosine_distance(v1, v2);
        return SQLITE_OK;
    } catch (...) {
        return SQLITE_ERROR;
    }
}

} // extern "C"
