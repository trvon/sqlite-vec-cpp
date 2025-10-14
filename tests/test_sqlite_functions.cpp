#include <sqlite3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::sqlite;

// RAII wrapper for SQLite database
class SQLiteDB {
public:
    SQLiteDB() {
        int rc = sqlite3_open(":memory:", &db_);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
        }
    }

    ~SQLiteDB() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    sqlite3* get() { return db_; }

    // Execute SQL and return result as string
    std::string exec_scalar(const std::string& sql) {
        char* err_msg = nullptr;
        std::string result;

        int rc = sqlite3_exec(
            db_, sql.c_str(),
            [](void* data, int argc, char** argv, char** col_names) -> int {
                if (argc > 0 && argv[0]) {
                    *static_cast<std::string*>(data) = argv[0];
                }
                return 0;
            },
            &result, &err_msg);

        if (rc != SQLITE_OK) {
            std::string error = err_msg ? err_msg : "Unknown error";
            sqlite3_free(err_msg);
            throw std::runtime_error("SQL error: " + error);
        }

        return result;
    }

    // Execute SQL without expecting result
    void exec(const std::string& sql) {
        char* err_msg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);

        if (rc != SQLITE_OK) {
            std::string error = err_msg ? err_msg : "Unknown error";
            sqlite3_free(err_msg);
            throw std::runtime_error("SQL error: " + error);
        }
    }

    // Register a function
    void create_function(const char* name, int nargs,
                         void (*func)(sqlite3_context*, int, sqlite3_value**),
                         int flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC) {
        int rc = sqlite3_create_function_v2(db_, name, nargs, flags, nullptr, func, nullptr,
                                            nullptr, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to create function: " + std::string(name));
        }
    }

private:
    sqlite3* db_ = nullptr;
};

// Default flags for functions that work with vector subtypes
constexpr int VEC_FUNC_FLAGS =
    SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE;

void register_all_functions(SQLiteDB& db) {
    // Distance functions (read subtypes, don't set them - return scalar)
    db.create_function("vec_distance_l2", 2, vec_distance_l2,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_l1", 2, vec_distance_l1,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_cosine", 2, vec_distance_cosine,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_distance_hamming", 2, vec_distance_hamming,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector creation (set subtypes)
    db.create_function("vec_f32", 1, vec_f32, VEC_FUNC_FLAGS);
    db.create_function("vec_int8", 1, vec_int8, VEC_FUNC_FLAGS);
    db.create_function("vec_bit", 1, vec_bit, VEC_FUNC_FLAGS);

    // Vector info (read subtypes)
    db.create_function("vec_length", 1, vec_length,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_type", 1, vec_type,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector conversion (read and set subtypes)
    db.create_function("vec_to_json", 1, vec_to_json,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);

    // Vector math (read and set subtypes)
    db.create_function("vec_add", 2, vec_add, VEC_FUNC_FLAGS);
    db.create_function("vec_sub", 2, vec_sub, VEC_FUNC_FLAGS);
    db.create_function("vec_normalize", 1, vec_normalize, VEC_FUNC_FLAGS);
    db.create_function("vec_slice", 3, vec_slice, VEC_FUNC_FLAGS);

    // Enhanced functions (read subtypes, may or may not set them)
    db.create_function("vec_dot", 2, vec_dot, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_magnitude", 1, vec_magnitude,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_scale", 2, vec_scale, VEC_FUNC_FLAGS);
    db.create_function("vec_mean", 1, vec_mean,
                       SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_std", 1, vec_std, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_min", 1, vec_min, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_max", 1, vec_max, SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_SUBTYPE);
    db.create_function("vec_clamp", 3, vec_clamp, VEC_FUNC_FLAGS);
}

// Helper to compare floats
bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// ============================================================================
// TEST: Vector Creation Functions
// ============================================================================

void test_vec_f32_json() {
    std::cout << "Testing vec_f32 with JSON input..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Create from JSON array
    std::string result = db.exec_scalar("SELECT hex(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(!result.empty());

    // Verify length
    std::string length = db.exec_scalar("SELECT vec_length(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(length == "3");

    // Verify type
    std::string type = db.exec_scalar("SELECT vec_type(vec_f32('[1.0, 2.0, 3.0]'))");
    assert(type == "float32");

    std::cout << "  ✓ vec_f32 JSON parsing works" << std::endl;
}

void test_vec_f32_blob() {
    std::cout << "Testing vec_f32 with blob input..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Create table with blob
    db.exec("CREATE TABLE test (vec BLOB)");

    // Insert binary data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db.get(), "INSERT INTO test VALUES (?)", -1, &stmt, nullptr);
    sqlite3_bind_blob(stmt, 1, data.data(), data.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    // Read back and verify
    std::string length = db.exec_scalar("SELECT vec_length(vec) FROM test");
    assert(length == "4");

    std::cout << "  ✓ vec_f32 blob handling works" << std::endl;
}

void test_vec_int8() {
    std::cout << "Testing vec_int8..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    std::string length = db.exec_scalar("SELECT vec_length(vec_int8('[1, 2, 3, 4, 5]'))");
    assert(length == "5");

    std::string type = db.exec_scalar("SELECT vec_type(vec_int8('[1, 2, 3]'))");
    assert(type == "int8");

    std::cout << "  ✓ vec_int8 works" << std::endl;
}

void test_vec_length() {
    std::cout << "Testing vec_length..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1]'))") == "1");
    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1,2,3]'))") == "3");
    assert(db.exec_scalar("SELECT vec_length(vec_f32('[1,2,3,4,5,6,7,8,9,10]'))") == "10");

    std::cout << "  ✓ vec_length works" << std::endl;
}

void test_vec_to_json() {
    std::cout << "Testing vec_to_json..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    std::string json = db.exec_scalar("SELECT vec_to_json(vec_f32('[1.0, 2.5, 3.14]'))");

    // Should be valid JSON array
    assert(json.front() == '[');
    assert(json.back() == ']');
    assert(json.find("1") != std::string::npos);
    assert(json.find("2.5") != std::string::npos);

    std::cout << "  ✓ vec_to_json works" << std::endl;
}

// ============================================================================
// TEST: Distance Functions
// ============================================================================

void test_distance_l2() {
    std::cout << "Testing vec_distance_l2..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Distance between [0,0] and [3,4] should be 5.0 (Euclidean distance, not squared)
    std::string dist = db.exec_scalar("SELECT vec_distance_l2(vec_f32('[0,0]'), vec_f32('[3,4]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 5.0)); // sqrt(3^2 + 4^2) = sqrt(25) = 5.0

    // Distance between identical vectors should be 0
    dist = db.exec_scalar("SELECT vec_distance_l2(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    std::cout << "  ✓ vec_distance_l2 works" << std::endl;
}

void test_distance_l1() {
    std::cout << "Testing vec_distance_l1..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Manhattan distance between [0,0] and [3,4] should be 7
    std::string dist = db.exec_scalar("SELECT vec_distance_l1(vec_f32('[0,0]'), vec_f32('[3,4]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 7.0));

    // Distance between identical vectors should be 0
    dist = db.exec_scalar("SELECT vec_distance_l1(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    std::cout << "  ✓ vec_distance_l1 works" << std::endl;
}

void test_distance_cosine() {
    std::cout << "Testing vec_distance_cosine..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Cosine distance between identical vectors should be 0
    std::string dist =
        db.exec_scalar("SELECT vec_distance_cosine(vec_f32('[1,2,3]'), vec_f32('[1,2,3]'))");
    double d = std::stod(dist);
    assert(approx_equal(d, 0.0));

    // Cosine distance between [1,0] and [0,1] should be 1.0 (orthogonal)
    dist = db.exec_scalar("SELECT vec_distance_cosine(vec_f32('[1,0]'), vec_f32('[0,1]'))");
    d = std::stod(dist);
    assert(approx_equal(d, 1.0));

    std::cout << "  ✓ vec_distance_cosine works" << std::endl;
}

// ============================================================================
// TEST: Vector Math Functions
// ============================================================================

void test_vec_add() {
    std::cout << "Testing vec_add..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_add(vec_f32('[1,2,3]'), vec_f32('[4,5,6]')))");

    // Result should be [5, 7, 9]
    assert(result.find("5") != std::string::npos);
    assert(result.find("7") != std::string::npos);
    assert(result.find("9") != std::string::npos);

    std::cout << "  ✓ vec_add works" << std::endl;
}

void test_vec_sub() {
    std::cout << "Testing vec_sub..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_sub(vec_f32('[5,7,9]'), vec_f32('[1,2,3]')))");

    // Result should be [4, 5, 6]
    assert(result.find("4") != std::string::npos);
    assert(result.find("5") != std::string::npos);
    assert(result.find("6") != std::string::npos);

    std::cout << "  ✓ vec_sub works" << std::endl;
}

void test_vec_normalize() {
    std::cout << "Testing vec_normalize..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Normalize [3, 4] -> [0.6, 0.8]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_normalize(vec_f32('[3,4]')))");

    assert(result.find("0.6") != std::string::npos);
    assert(result.find("0.8") != std::string::npos);

    // Verify magnitude is 1
    std::string mag = db.exec_scalar("SELECT vec_magnitude(vec_normalize(vec_f32('[3,4]')))");
    double m = std::stod(mag);
    assert(approx_equal(m, 1.0));

    std::cout << "  ✓ vec_normalize works" << std::endl;
}

void test_vec_slice() {
    std::cout << "Testing vec_slice..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Slice [1,2,3,4,5] from index 1 to 4 -> [2,3,4]
    std::string length =
        db.exec_scalar("SELECT vec_length(vec_slice(vec_f32('[1,2,3,4,5]'), 1, 4))");
    assert(length == "3");

    std::string result =
        db.exec_scalar("SELECT vec_to_json(vec_slice(vec_f32('[1,2,3,4,5]'), 1, 4))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("3") != std::string::npos);
    assert(result.find("4") != std::string::npos);

    std::cout << "  ✓ vec_slice works" << std::endl;
}

// ============================================================================
// TEST: Enhanced Functions
// ============================================================================

void test_vec_dot() {
    std::cout << "Testing vec_dot..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Dot product of [1,2,3] and [4,5,6] = 1*4 + 2*5 + 3*6 = 32
    std::string result = db.exec_scalar("SELECT vec_dot(vec_f32('[1,2,3]'), vec_f32('[4,5,6]'))");
    double dot = std::stod(result);
    assert(approx_equal(dot, 32.0));

    // Dot product of orthogonal vectors should be 0
    result = db.exec_scalar("SELECT vec_dot(vec_f32('[1,0]'), vec_f32('[0,1]'))");
    dot = std::stod(result);
    assert(approx_equal(dot, 0.0));

    std::cout << "  ✓ vec_dot works" << std::endl;
}

void test_vec_magnitude() {
    std::cout << "Testing vec_magnitude..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Magnitude of [3,4] should be 5
    std::string result = db.exec_scalar("SELECT vec_magnitude(vec_f32('[3,4]'))");
    double mag = std::stod(result);
    assert(approx_equal(mag, 5.0));

    // Magnitude of [1,0,0] should be 1
    result = db.exec_scalar("SELECT vec_magnitude(vec_f32('[1,0,0]'))");
    mag = std::stod(result);
    assert(approx_equal(mag, 1.0));

    std::cout << "  ✓ vec_magnitude works" << std::endl;
}

void test_vec_scale() {
    std::cout << "Testing vec_scale..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Scale [1,2,3] by 2 -> [2,4,6]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_scale(vec_f32('[1,2,3]'), 2.0))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("4") != std::string::npos);
    assert(result.find("6") != std::string::npos);

    std::cout << "  ✓ vec_scale works" << std::endl;
}

void test_vec_mean() {
    std::cout << "Testing vec_mean..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Mean of [1,2,3,4,5] should be 3
    std::string result = db.exec_scalar("SELECT vec_mean(vec_f32('[1,2,3,4,5]'))");
    double mean = std::stod(result);
    assert(approx_equal(mean, 3.0));

    // Mean of [10,20,30] should be 20
    result = db.exec_scalar("SELECT vec_mean(vec_f32('[10,20,30]'))");
    mean = std::stod(result);
    assert(approx_equal(mean, 20.0));

    std::cout << "  ✓ vec_mean works" << std::endl;
}

void test_vec_std() {
    std::cout << "Testing vec_std..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Standard deviation should be > 0 for varying data
    std::string result = db.exec_scalar("SELECT vec_std(vec_f32('[1,2,3,4,5]'))");
    double std_dev = std::stod(result);
    assert(std_dev > 0.0);
    assert(approx_equal(std_dev, 1.5811, 0.001)); // Sample std dev

    std::cout << "  ✓ vec_std works" << std::endl;
}

void test_vec_min_max() {
    std::cout << "Testing vec_min and vec_max..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    std::string min_val = db.exec_scalar("SELECT vec_min(vec_f32('[3,1,4,1,5,9,2,6]'))");
    double min_d = std::stod(min_val);
    assert(approx_equal(min_d, 1.0));

    std::string max_val = db.exec_scalar("SELECT vec_max(vec_f32('[3,1,4,1,5,9,2,6]'))");
    double max_d = std::stod(max_val);
    assert(approx_equal(max_d, 9.0));

    std::cout << "  ✓ vec_min and vec_max work" << std::endl;
}

void test_vec_clamp() {
    std::cout << "Testing vec_clamp..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Clamp [1,5,10] to [2,8] -> [2,5,8]
    std::string result = db.exec_scalar("SELECT vec_to_json(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");

    assert(result.find("2") != std::string::npos);
    assert(result.find("5") != std::string::npos);
    assert(result.find("8") != std::string::npos);

    // Verify min is 2
    std::string min_val = db.exec_scalar("SELECT vec_min(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");
    double min_d = std::stod(min_val);
    assert(approx_equal(min_d, 2.0));

    // Verify max is 8
    std::string max_val = db.exec_scalar("SELECT vec_max(vec_clamp(vec_f32('[1,5,10]'), 2, 8))");
    double max_d = std::stod(max_val);
    assert(approx_equal(max_d, 8.0));

    std::cout << "  ✓ vec_clamp works" << std::endl;
}

// ============================================================================
// TEST: Error Handling
// ============================================================================

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Dimension mismatch
    try {
        db.exec_scalar("SELECT vec_distance_l2(vec_f32('[1,2]'), vec_f32('[1,2,3]'))");
        assert(false && "Should have thrown error for dimension mismatch");
    } catch (const std::runtime_error& e) {
        // Expected
        std::string err = e.what();
        assert(err.find("mismatch") != std::string::npos ||
               err.find("SQL error") != std::string::npos);
    }

    // Invalid JSON
    try {
        db.exec_scalar("SELECT vec_f32('[1,2,invalid]')");
        assert(false && "Should have thrown error for invalid JSON");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Empty vector
    try {
        db.exec_scalar("SELECT vec_f32('[]')");
        assert(false && "Should have thrown error for empty vector");
    } catch (const std::runtime_error&) {
        // Expected
    }

    std::cout << "  ✓ Error handling works" << std::endl;
}

// ============================================================================
// TEST: Integration Scenarios
// ============================================================================

void test_integration_similarity_search() {
    std::cout << "Testing integration: similarity search..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Create table with vectors
    db.exec("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding BLOB)");

    // Insert vectors using vec_f32
    db.exec("INSERT INTO documents VALUES (1, vec_f32('[1,0,0]'))");
    db.exec("INSERT INTO documents VALUES (2, vec_f32('[0,1,0]'))");
    db.exec("INSERT INTO documents VALUES (3, vec_f32('[1,1,0]'))");

    // Find document closest to [1,0.5,0]
    // Documents 1 and 3 are tied at distance 0.5, so either is correct
    std::string closest =
        db.exec_scalar("SELECT id FROM documents "
                       "ORDER BY vec_distance_l2(embedding, vec_f32('[1,0.5,0]')) "
                       "LIMIT 1");

    // Should be document 1 or 3 (both at distance 0.5)
    assert(closest == "1" || closest == "3");

    std::cout << "  ✓ Similarity search integration works" << std::endl;
}

void test_integration_vector_operations() {
    std::cout << "Testing integration: vector operations pipeline..." << std::endl;

    SQLiteDB db;
    register_all_functions(db);

    // Complex pipeline: create, normalize, scale, and compute distance
    std::string result = db.exec_scalar("SELECT vec_distance_l2("
                                        "  vec_scale(vec_normalize(vec_f32('[3,4]')), 5),"
                                        "  vec_f32('[3,4]')"
                                        ")");

    double dist = std::stod(result);
    assert(dist >= 0.0); // Should be valid distance

    // Chain math operations
    result = db.exec_scalar("SELECT vec_to_json("
                            "  vec_add("
                            "    vec_sub(vec_f32('[10,10,10]'), vec_f32('[5,5,5]')),"
                            "    vec_f32('[1,2,3]')"
                            "  )"
                            ")");

    // Should be [6,7,8]
    assert(result.find("6") != std::string::npos);
    assert(result.find("7") != std::string::npos);
    assert(result.find("8") != std::string::npos);

    std::cout << "  ✓ Vector operations pipeline works" << std::endl;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "SQLITE-VEC-CPP COMPREHENSIVE TEST SUITE\n";
    std::cout << "========================================\n\n";

    int tests_passed = 0;
    int tests_failed = 0;

    auto run_test = [&](const char* name, void (*test_func)()) {
        try {
            test_func();
            tests_passed++;
        } catch (const std::exception& e) {
            std::cerr << "\n✗ TEST FAILED: " << name << "\n";
            std::cerr << "  Error: " << e.what() << "\n";
            tests_failed++;
        } catch (...) {
            std::cerr << "\n✗ TEST FAILED: " << name << "\n";
            std::cerr << "  Unknown error\n";
            tests_failed++;
        }
    };

    std::cout << "Testing Vector Creation Functions:\n";
    run_test("vec_f32_json", test_vec_f32_json);
    run_test("vec_f32_blob", test_vec_f32_blob);
    run_test("vec_int8", test_vec_int8);
    run_test("vec_length", test_vec_length);
    run_test("vec_to_json", test_vec_to_json);

    std::cout << "\nTesting Distance Functions:\n";
    run_test("distance_l2", test_distance_l2);
    run_test("distance_l1", test_distance_l1);
    run_test("distance_cosine", test_distance_cosine);

    std::cout << "\nTesting Vector Math Functions:\n";
    run_test("vec_add", test_vec_add);
    run_test("vec_sub", test_vec_sub);
    run_test("vec_normalize", test_vec_normalize);
    run_test("vec_slice", test_vec_slice);

    std::cout << "\nTesting Enhanced Functions:\n";
    run_test("vec_dot", test_vec_dot);
    run_test("vec_magnitude", test_vec_magnitude);
    run_test("vec_scale", test_vec_scale);
    run_test("vec_mean", test_vec_mean);
    run_test("vec_std", test_vec_std);
    run_test("vec_min_max", test_vec_min_max);
    run_test("vec_clamp", test_vec_clamp);

    std::cout << "\nTesting Error Handling:\n";
    run_test("error_handling", test_error_handling);

    std::cout << "\nTesting Integration Scenarios:\n";
    run_test("integration_similarity_search", test_integration_similarity_search);
    run_test("integration_vector_operations", test_integration_vector_operations);

    std::cout << "\n========================================\n";
    std::cout << "TEST RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "Total tests:  " << (tests_passed + tests_failed) << "\n";

    if (tests_failed == 0) {
        std::cout << "\n✓ All tests passed!\n\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed\n\n";
        return 1;
    }
}
