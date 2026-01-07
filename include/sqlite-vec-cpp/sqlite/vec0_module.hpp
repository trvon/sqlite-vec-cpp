#pragma once

#include <sqlite3.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "../utils/error.hpp"
#include "parsers.hpp"
#include "value.hpp"

namespace sqlite_vec_cpp::sqlite {

// Vec0 Virtual Table Implementation
// Provides full CREATE VIRTUAL TABLE ... USING vec0(...) support
// Compatible with original sqlite-vec C implementation

struct Vec0Table {
    sqlite3_vtab base;
    sqlite3* db;
    std::string schema_name;
    std::string table_name;
    std::string embedding_column;
    size_t dimensions;
    bool use_shadow_tables;

    Vec0Table() : db(nullptr), dimensions(384), use_shadow_tables(true) {
        std::memset(&base, 0, sizeof(base));
    }
};

struct Vec0Cursor {
    sqlite3_vtab_cursor base;
    Vec0Table* table;
    sqlite3_stmt* stmt; // For shadow table queries
    int64_t current_rowid;
    bool eof;

    Vec0Cursor() : table(nullptr), stmt(nullptr), current_rowid(0), eof(true) {
        std::memset(&base, 0, sizeof(base));
    }

    ~Vec0Cursor() {
        if (stmt) {
            sqlite3_finalize(stmt);
            stmt = nullptr;
        }
    }
};

// Helper: Parse CREATE VIRTUAL TABLE statement for embedding column and dimensions
inline bool parse_vec0_schema(int argc, const char* const* argv, std::string& embedding_col,
                              size_t& dims) {
    // Format: CREATE VIRTUAL TABLE name USING vec0(embedding float[dims])
    embedding_col = "embedding"; // default
    dims = 384;                  // default

    for (int i = 3; i < argc; i++) {
        std::string arg(argv[i]);

        // Look for pattern: column_name float[dimensions]
        auto float_pos = arg.find("float[");
        if (float_pos != std::string::npos) {
            // Extract column name (everything before "float[")
            embedding_col = arg.substr(0, float_pos);
            // Trim whitespace
            while (!embedding_col.empty() && std::isspace(embedding_col.back())) {
                embedding_col.pop_back();
            }

            // Extract dimensions
            auto end_bracket = arg.find(']', float_pos);
            if (end_bracket != std::string::npos) {
                try {
                    std::string dim_str = arg.substr(float_pos + 6, end_bracket - float_pos - 6);
                    dims = std::stoul(dim_str);
                    return true;
                } catch (...) {
                }
            }
        }
    }
    return false;
}

// Helper: Create shadow tables for vec0 storage
inline int create_shadow_tables(sqlite3* db, const char* schema, const char* table,
                                const char* embedding_col, [[maybe_unused]] size_t dims,
                                char** pzErr) {
    // Note: dims parameter is reserved for future use (validation, typed vectors)
    (void)dims;

    // Shadow table for metadata
    std::ostringstream meta_sql;
    meta_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_metadata\" ("
             << "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
             << "data BLOB)";

    char* err = nullptr;
    int rc = sqlite3_exec(db, meta_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create metadata shadow table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    // Shadow table for vectors
    std::ostringstream vec_sql;
    vec_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_vectors\" ("
            << "rowid INTEGER PRIMARY KEY, "
            << "\"" << embedding_col << "\" BLOB)";

    rc = sqlite3_exec(db, vec_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create vectors shadow table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    return SQLITE_OK;
}

// xCreate: Called when CREATE VIRTUAL TABLE is executed
inline int vec0Create(sqlite3* db, void* pAux, int argc, const char* const* argv,
                      sqlite3_vtab** ppVTab, char** pzErr) {
    (void)pAux;

    // Enable shadow table writes through xUpdate
    sqlite3_vtab_config(db, SQLITE_VTAB_DIRECTONLY);

    if (argc < 3) {
        *pzErr = sqlite3_mprintf("vec0: insufficient arguments");
        return SQLITE_ERROR;
    }

    std::string embedding_col;
    size_t dims;
    parse_vec0_schema(argc, argv, embedding_col, dims);

    auto* table = new Vec0Table();
    table->db = db;
    table->schema_name = argv[1];
    table->table_name = argv[2];
    table->embedding_column = embedding_col;
    table->dimensions = dims;

    // Create shadow tables for actual storage
    int rc = create_shadow_tables(db, argv[1], argv[2], embedding_col.c_str(), dims, pzErr);
    if (rc != SQLITE_OK) {
        delete table;
        return rc;
    }

    // Declare virtual table schema
    std::ostringstream schema;
    schema << "CREATE TABLE x(rowid INTEGER PRIMARY KEY, "
           << "\"" << embedding_col << "\")";

    rc = sqlite3_declare_vtab(db, schema.str().c_str());
    if (rc != SQLITE_OK) {
        delete table;
        *pzErr = sqlite3_mprintf("Failed to declare vtab schema");
        return rc;
    }

    *ppVTab = &table->base;
    return SQLITE_OK;
}

// xConnect: Called when connecting to existing virtual table
inline int vec0Connect(sqlite3* db, void* pAux, int argc, const char* const* argv,
                       sqlite3_vtab** ppVTab, char** pzErr) {
    // For vec0, Connect behaves same as Create (shadow tables already exist)
    return vec0Create(db, pAux, argc, argv, ppVTab, pzErr);
}

// xDisconnect: Called when disconnecting from table
inline int vec0Disconnect(sqlite3_vtab* pVTab) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);
    delete table;
    return SQLITE_OK;
}

// xDestroy: Called when DROP TABLE is executed
inline int vec0Destroy(sqlite3_vtab* pVTab) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);

    // Drop shadow tables
    std::ostringstream drop_meta;
    drop_meta << "DROP TABLE IF EXISTS \"" << table->schema_name << "\".\"" << table->table_name
              << "_metadata\"";
    sqlite3_exec(table->db, drop_meta.str().c_str(), nullptr, nullptr, nullptr);

    std::ostringstream drop_vec;
    drop_vec << "DROP TABLE IF EXISTS \"" << table->schema_name << "\".\"" << table->table_name
             << "_vectors\"";
    sqlite3_exec(table->db, drop_vec.str().c_str(), nullptr, nullptr, nullptr);

    delete table;
    return SQLITE_OK;
}

// xBestIndex: Query planner callback
inline int vec0BestIndex(sqlite3_vtab* pVTab, sqlite3_index_info* pInfo) {
    (void)pVTab;

    // Simple implementation: full table scan
    pInfo->estimatedCost = 1000000.0;
    pInfo->estimatedRows = 1000000;
    pInfo->idxNum = 0;

    return SQLITE_OK;
}

// xOpen: Create a new cursor
inline int vec0Open(sqlite3_vtab* pVTab, sqlite3_vtab_cursor** ppCursor) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);
    auto* cursor = new Vec0Cursor();
    cursor->table = table;
    *ppCursor = &cursor->base;
    return SQLITE_OK;
}

// xClose: Destroy cursor
inline int vec0Close(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    delete cursor;
    return SQLITE_OK;
}

// xFilter: Begin iteration
inline int vec0Filter(sqlite3_vtab_cursor* pCursor, int idxNum, const char* idxStr, int argc,
                      sqlite3_value** argv) {
    (void)idxNum;
    (void)idxStr;
    (void)argc;
    (void)argv;

    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    auto* table = cursor->table;

    // Prepare query on shadow table
    if (cursor->stmt) {
        sqlite3_finalize(cursor->stmt);
        cursor->stmt = nullptr;
    }

    std::ostringstream sql;
    sql << "SELECT rowid, \"" << table->embedding_column << "\" FROM \"" << table->schema_name
        << "\".\"" << table->table_name << "_vectors\" ORDER BY rowid";

    int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &cursor->stmt, nullptr);
    if (rc != SQLITE_OK) {
        cursor->eof = true;
        return rc;
    }

    // Advance to first row
    rc = sqlite3_step(cursor->stmt);
    if (rc == SQLITE_ROW) {
        cursor->current_rowid = sqlite3_column_int64(cursor->stmt, 0);
        cursor->eof = false;
    } else {
        cursor->eof = true;
    }

    return SQLITE_OK;
}

// xNext: Advance to next row
inline int vec0Next(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);

    if (!cursor->stmt || cursor->eof) {
        return SQLITE_OK;
    }

    int rc = sqlite3_step(cursor->stmt);
    if (rc == SQLITE_ROW) {
        cursor->current_rowid = sqlite3_column_int64(cursor->stmt, 0);
        cursor->eof = false;
    } else {
        cursor->eof = true;
    }

    return SQLITE_OK;
}

// xEof: Check if at end of result set
inline int vec0Eof(sqlite3_vtab_cursor* pCursor) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    return cursor->eof ? 1 : 0;
}

// xColumn: Return column value
inline int vec0Column(sqlite3_vtab_cursor* pCursor, sqlite3_context* ctx, int col) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);

    if (!cursor->stmt || cursor->eof) {
        sqlite3_result_null(ctx);
        return SQLITE_OK;
    }

    if (col == 0) {
        // rowid column
        sqlite3_result_int64(ctx, cursor->current_rowid);
    } else if (col == 1) {
        // embedding column
        const void* blob = sqlite3_column_blob(cursor->stmt, 1);
        int bytes = sqlite3_column_bytes(cursor->stmt, 1);
        if (blob && bytes > 0) {
            sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
        } else {
            sqlite3_result_null(ctx);
        }
    } else {
        sqlite3_result_null(ctx);
    }

    return SQLITE_OK;
}

// xRowid: Return current rowid
inline int vec0Rowid(sqlite3_vtab_cursor* pCursor, sqlite3_int64* pRowid) {
    auto* cursor = reinterpret_cast<Vec0Cursor*>(pCursor);
    *pRowid = cursor->current_rowid;
    return SQLITE_OK;
}

// xUpdate: INSERT, UPDATE, DELETE operations
inline int vec0Update(sqlite3_vtab* pVTab, int argc, sqlite3_value** argv, sqlite3_int64* pRowid) {
    auto* table = reinterpret_cast<Vec0Table*>(pVTab);

    // DELETE: argc==1, argv[0]=old_rowid
    if (argc == 1) {
        int64_t rowid = sqlite3_value_int64(argv[0]);
        std::ostringstream sql;
        sql << "DELETE FROM \"" << table->schema_name << "\".\"" << table->table_name
            << "_vectors\" WHERE rowid=" << rowid;
        return sqlite3_exec(table->db, sql.str().c_str(), nullptr, nullptr, nullptr);
    }

    // INSERT: argc>1, argv[0]=NULL, argv[1]=new_rowid or NULL
    // UPDATE: argc>1, argv[0]=old_rowid, argv[1]=new_rowid
    if (argc > 1) {
        bool is_insert = (sqlite3_value_type(argv[0]) == SQLITE_NULL);
        int64_t rowid = is_insert ? 0 : sqlite3_value_int64(argv[0]);

        // argv layout per SQLite vtab spec:
        // - argv[0]: old rowid (or NULL)
        // - argv[1]: new rowid (or NULL)
        // - argv[2..]: column values in declared order
        if (argc < 4) {
            return SQLITE_MISUSE;
        }

        // Our declared schema is: CREATE TABLE x(rowid INTEGER PRIMARY KEY, "embedding")
        // So the embedding column value is argv[3] (argv[2] corresponds to rowid column value).
        sqlite3_value* embedding_val = argv[3];
        const void* blob = sqlite3_value_blob(embedding_val);
        int bytes = sqlite3_value_bytes(embedding_val);

        // If xUpdate doesn't materialize function-returned blobs, parse JSON/text and bind as blob
        std::vector<float> parsed_vec;
        if (sqlite3_value_type(embedding_val) == SQLITE_TEXT) {
            Value value(embedding_val);
            auto parsed = parse_vector_from_value<float>(value);
            if (parsed) {
                parsed_vec = std::move(parsed.value());
                blob = parsed_vec.data();
                bytes = static_cast<int>(parsed_vec.size() * sizeof(float));
            }
        }

        if (is_insert) {
            std::ostringstream sql;
            sql << "INSERT INTO \"" << table->schema_name << "\".\"" << table->table_name
                << "_vectors\" (\"" << table->embedding_column << "\") VALUES (?)";

            sqlite3_stmt* stmt;
            int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK)
                return rc;

            if (blob && bytes > 0) {
                sqlite3_bind_blob(stmt, 1, blob, bytes, SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(stmt, 1);
            }

            rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                *pRowid = sqlite3_last_insert_rowid(table->db);
                rc = SQLITE_OK;
            }

            sqlite3_finalize(stmt);
            return rc;
        } else {
            std::ostringstream sql;
            sql << "UPDATE \"" << table->schema_name << "\".\"" << table->table_name
                << "_vectors\" SET \"" << table->embedding_column << "\"=? WHERE rowid=" << rowid;

            sqlite3_stmt* stmt;
            int rc = sqlite3_prepare_v2(table->db, sql.str().c_str(), -1, &stmt, nullptr);
            if (rc != SQLITE_OK)
                return rc;

            if (blob && bytes > 0) {
                sqlite3_bind_blob(stmt, 1, blob, bytes, SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(stmt, 1);
            }

            rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                *pRowid = rowid;
                rc = SQLITE_OK;
            }

            sqlite3_finalize(stmt);
            return rc;
        }
    }

    return SQLITE_OK;
}

// Module definition
// Note: sqlite3_module struct size varies by SQLite version:
// - xShadowName added in 3.26.0 (20 fields with iVersion=2)
// - xIntegrity added in 3.44.0 (21 fields with iVersion=3)
// We use designated initializers for forward compatibility
static sqlite3_module vec0_module = {
    /* iVersion      */ 2,
    /* xCreate       */ vec0Create,
    /* xConnect      */ vec0Connect,
    /* xBestIndex    */ vec0BestIndex,
    /* xDisconnect   */ vec0Disconnect,
    /* xDestroy      */ vec0Destroy,
    /* xOpen         */ vec0Open,
    /* xClose        */ vec0Close,
    /* xFilter       */ vec0Filter,
    /* xNext         */ vec0Next,
    /* xEof          */ vec0Eof,
    /* xColumn       */ vec0Column,
    /* xRowid        */ vec0Rowid,
    /* xUpdate       */ vec0Update,
    /* xBegin        */ nullptr,
    /* xSync         */ nullptr,
    /* xCommit       */ nullptr,
    /* xRollback     */ nullptr,
    /* xFindFunction */ nullptr,
    /* xRename       */ nullptr,
    /* xSavepoint    */ nullptr,
    /* xRelease      */ nullptr,
    /* xRollbackTo   */ nullptr,
    /* xShadowName   */ nullptr,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity    */ nullptr,
#endif
};

inline Result<void> register_vec0_module(sqlite3* db) {
    if (!db) {
        return err<void>(Error::invalid_argument("database handle is null"));
    }

    int rc = sqlite3_create_module_v2(db, "vec0", &vec0_module, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        return err<void>(Error::sqlite_error("Failed to register vec0 module", rc));
    }

    return Result<void>();
}

} // namespace sqlite_vec_cpp::sqlite
