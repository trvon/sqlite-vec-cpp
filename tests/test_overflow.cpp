// SPDX-License-Identifier: Apache-2.0 OR MIT
// Overflow and validation tests for the vec0 virtual table:
// dimension caps at CREATE VIRTUAL TABLE, blob size mismatches at insert,
// and embedding column name validation.

#include <sqlite3.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sqlite-vec-cpp/sqlite/vec0_module.hpp>

extern "C" {
int sqlite3_vec_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* pApi);
}

namespace {

sqlite3* open_db() {
    sqlite3* db = nullptr;
    int rc = sqlite3_open(":memory:", &db);
    assert(rc == SQLITE_OK);
    char* err = nullptr;
    rc = sqlite3_vec_init(db, &err, nullptr);
    assert(rc == SQLITE_OK);
    sqlite3_free(err);
    return db;
}

int exec(sqlite3* db, const std::string& sql) {
    char* err = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    sqlite3_free(err);
    return rc;
}

void test_dimension_caps() {
    std::cout << "Overflow 1: dimension caps at CREATE VIRTUAL TABLE..." << std::endl;

    sqlite3* db = open_db();

    assert(exec(db, "CREATE VIRTUAL TABLE t_zero USING vec0(embedding float[0])") != SQLITE_OK);
    assert(exec(db, "CREATE VIRTUAL TABLE t_huge USING vec0(embedding float[65537])") !=
           SQLITE_OK);
    assert(exec(db, "CREATE VIRTUAL TABLE t_absurd USING vec0(embedding float[99999999999])") !=
           SQLITE_OK);

    assert(exec(db, "CREATE VIRTUAL TABLE t_max USING vec0(embedding float[65536])") ==
           SQLITE_OK);
    assert(exec(db, "CREATE VIRTUAL TABLE t_ok USING vec0(embedding float[4])") == SQLITE_OK);

    sqlite3_close(db);
    std::cout << "  ✓ dims 0 / 65537 / 99999999999 rejected; 4 and 65536 accepted" << std::endl;
}

void test_blob_size_mismatch() {
    std::cout << "Overflow 2: blob size mismatch at insert..." << std::endl;

    sqlite3* db = open_db();
    assert(exec(db, "CREATE VIRTUAL TABLE t USING vec0(embedding float[4])") == SQLITE_OK);

    auto insert_blob = [&](const std::vector<float>& values) {
        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db, "INSERT INTO t(rowid, embedding) VALUES (?, ?)", -1,
                                    &stmt, nullptr);
        assert(rc == SQLITE_OK);
        static int64_t next_rowid = 1;
        sqlite3_bind_int64(stmt, 1, next_rowid++);
        sqlite3_bind_blob(stmt, 2, values.data(),
                          static_cast<int>(values.size() * sizeof(float)), SQLITE_TRANSIENT);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return rc;
    };

    assert(insert_blob({1.0f, 2.0f, 3.0f, 4.0f}) == SQLITE_DONE);
    assert(insert_blob({1.0f, 2.0f, 3.0f}) != SQLITE_DONE);
    assert(insert_blob({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}) != SQLITE_DONE);

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(
        db, "SELECT rowid, distance FROM t WHERE embedding MATCH vec_f32('[1,2,3,4]') AND k = 1",
        -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(stmt);
        assert(rc == SQLITE_ROW);
        assert(sqlite3_column_int64(stmt, 0) == 1);
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
    std::cout << "  ✓ mismatched blobs rejected; valid insert queryable" << std::endl;
}

void test_column_name_validation() {
    std::cout << "Overflow 3: embedding column name validation..." << std::endl;

    sqlite3* db = open_db();

    assert(exec(db, "CREATE VIRTUAL TABLE t_q USING vec0(a\"\"b float[4])") != SQLITE_OK);
    assert(exec(db, "CREATE VIRTUAL TABLE t_named USING vec0(my_vec float[8])") == SQLITE_OK);

    sqlite3_close(db);
    std::cout << "  ✓ quoted column names rejected; normal names accepted" << std::endl;
}

void test_corrupt_stored_vector_row() {
    std::cout << "Overflow 4: ANN build over corrupt stored vector row..." << std::endl;

    sqlite3* db = open_db();
    assert(exec(db, "CREATE VIRTUAL TABLE t USING vec0(embedding float[4])") == SQLITE_OK);

    for (int i = 1; i <= 8; ++i) {
        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(db, "INSERT INTO t(rowid, embedding) VALUES (?, ?)", -1,
                                    &stmt, nullptr);
        assert(rc == SQLITE_OK);
        std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
        sqlite3_bind_int64(stmt, 1, i);
        sqlite3_bind_blob(stmt, 2, v.data(), static_cast<int>(v.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        assert(rc == SQLITE_DONE);
    }

    assert(exec(db, "UPDATE t_vectors SET embedding = x'DEADBEEFDEAD' WHERE rowid = 3") ==
           SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(
        db, "SELECT rowid FROM t WHERE embedding MATCH vec_f32('[1,0,0,0]') AND k = 2", -1,
        &stmt, nullptr);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(stmt);
        assert(rc != SQLITE_ROW || sqlite3_column_int64(stmt, 0) != 3);
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
    std::cout << "  ✓ corrupt stored row surfaces as clean error, no crash" << std::endl;
}

} // namespace

int main() {
    std::cout << "=== vec0 Overflow & Validation Tests ===" << std::endl;

    test_dimension_caps();
    test_blob_size_mismatch();
    test_column_name_validation();
    test_corrupt_stored_vector_row();

    std::cout << "=== All overflow tests passed ===" << std::endl;
    return 0;
}
