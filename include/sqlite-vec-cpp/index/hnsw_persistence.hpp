#pragma once

#include <sqlite3.h>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "hnsw.hpp"
#include "hnsw_node.hpp"

namespace sqlite_vec_cpp::index {

/// HNSW Index Persistence Layer
/// Provides serialization/deserialization to SQLite shadow tables
///
/// Shadow Table Schema:
/// - {table}_hnsw_meta: Stores index metadata (config, entry point, etc.)
/// - {table}_hnsw_nodes: Stores HNSW graph structure (nodes + edges)

/// Serialize HNSW index configuration to blob
template <typename T, typename Metric>
std::vector<uint8_t> serialize_hnsw_config(const typename HNSWIndex<T, Metric>::Config& config) {
    std::vector<uint8_t> blob;
    blob.reserve(64);

    // Version marker (for future compatibility)
    constexpr uint32_t version = 1;
    auto write_u32 = [&](uint32_t val) {
        for (int i = 0; i < 4; ++i) {
            blob.push_back((val >> (i * 8)) & 0xFF);
        }
    };
    auto write_u64 = [&](uint64_t val) {
        for (int i = 0; i < 8; ++i) {
            blob.push_back((val >> (i * 8)) & 0xFF);
        }
    };
    auto write_f32 = [&](float val) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        write_u32(bits);
    };

    write_u32(version);
    write_u64(config.M);
    write_u64(config.M_max);
    write_u64(config.M_max_0);
    write_u64(config.ef_construction);
    write_f32(config.ml_factor);

    return blob;
}

/// Deserialize HNSW index configuration from blob
template <typename T, typename Metric>
typename HNSWIndex<T, Metric>::Config deserialize_hnsw_config(const void* blob, size_t size) {
    // Config blob: version(4) + M(8) + M_max(8) + M_max_0(8) + ef_construction(8) + ml_factor(4) =
    // 40 bytes
    if (size < 40) {
        throw std::runtime_error("Invalid HNSW config blob: too small");
    }

    const uint8_t* data = static_cast<const uint8_t*>(blob);
    size_t offset = 0;

    auto read_u32 = [&]() -> uint32_t {
        uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            val |= (static_cast<uint32_t>(data[offset++]) << (i * 8));
        }
        return val;
    };
    auto read_u64 = [&]() -> uint64_t {
        uint64_t val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<uint64_t>(data[offset++]) << (i * 8));
        }
        return val;
    };
    auto read_f32 = [&]() -> float {
        uint32_t bits = read_u32();
        float val;
        std::memcpy(&val, &bits, sizeof(float));
        return val;
    };

    uint32_t version = read_u32();
    if (version != 1) {
        throw std::runtime_error("Unsupported HNSW config version");
    }

    typename HNSWIndex<T, Metric>::Config config;
    config.M = read_u64();
    config.M_max = read_u64();
    config.M_max_0 = read_u64();
    config.ef_construction = read_u64();
    config.ml_factor = read_f32();

    return config;
}

/// Serialize HNSW node to blob
template <typename T> std::vector<uint8_t> serialize_hnsw_node(const HNSWNode<T>& node) {
    std::vector<uint8_t> blob;
    blob.reserve(1024); // Typical node size

    auto write_u64 = [&](uint64_t val) {
        for (int i = 0; i < 8; ++i) {
            blob.push_back((val >> (i * 8)) & 0xFF);
        }
    };

    // Write node ID
    write_u64(node.id);

    // Write vector dimensions
    write_u64(node.vector.size());

    // Write vector data (as bytes, type-specific)
    const uint8_t* vec_bytes = reinterpret_cast<const uint8_t*>(node.vector.data());
    size_t vec_bytes_size = node.vector.size() * sizeof(T);
    blob.insert(blob.end(), vec_bytes, vec_bytes + vec_bytes_size);

    // Write number of layers
    write_u64(node.edges.size());

    // Write edges for each layer
    for (const auto& layer : node.edges) {
        write_u64(layer.size()); // Number of neighbors at this layer
        for (size_t neighbor_id : layer) {
            write_u64(neighbor_id);
        }
    }

    return blob;
}

/// Deserialize HNSW node from blob
template <typename T> HNSWNode<T> deserialize_hnsw_node(const void* blob, size_t size) {
    const uint8_t* data = static_cast<const uint8_t*>(blob);
    size_t offset = 0;

    auto read_u64 = [&]() -> uint64_t {
        if (offset + 8 > size)
            throw std::runtime_error("HNSW node blob truncated");
        uint64_t val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<uint64_t>(data[offset++]) << (i * 8));
        }
        return val;
    };

    // Read node ID
    size_t node_id = read_u64();

    // Read vector dimensions
    size_t dim = read_u64();

    // Read vector data
    if (offset + dim * sizeof(T) > size) {
        throw std::runtime_error("HNSW node vector data truncated");
    }
    std::vector<T> vector(dim);
    std::memcpy(vector.data(), data + offset, dim * sizeof(T));
    offset += dim * sizeof(T);

    // Read number of layers
    size_t num_layers = read_u64();

    // Create node with placeholder layer count
    HNSWNode<T> node(node_id, std::span<const T>{vector}, num_layers > 0 ? num_layers - 1 : 0);

    // Read edges for each layer
    for (size_t layer = 0; layer < num_layers; ++layer) {
        size_t num_neighbors = read_u64();
        for (size_t i = 0; i < num_neighbors; ++i) {
            size_t neighbor_id = read_u64();
            // Edges are already allocated, so directly populate
            node.edges[layer].push_back(neighbor_id);
        }
    }

    return node;
}

/// Create HNSW shadow tables for persistence
inline int create_hnsw_shadow_tables(sqlite3* db, const char* schema, const char* table,
                                     char** pzErr) {
    // Metadata table (config, entry point, layer)
    std::ostringstream meta_sql;
    meta_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_hnsw_meta\" ("
             << "key TEXT PRIMARY KEY, "
             << "value BLOB)";

    char* err = nullptr;
    int rc = sqlite3_exec(db, meta_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create HNSW meta table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    // Nodes table (graph structure)
    std::ostringstream nodes_sql;
    nodes_sql << "CREATE TABLE IF NOT EXISTS \"" << schema << "\".\"" << table << "_hnsw_nodes\" ("
              << "node_id INTEGER PRIMARY KEY, "
              << "data BLOB NOT NULL)";

    rc = sqlite3_exec(db, nodes_sql.str().c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (pzErr && err) {
            *pzErr = sqlite3_mprintf("Failed to create HNSW nodes table: %s", err);
        }
        sqlite3_free(err);
        return rc;
    }

    return SQLITE_OK;
}

/// Save HNSW index to SQLite shadow tables
template <typename T, typename Metric>
int save_hnsw_index(sqlite3* db, const char* schema, const char* table,
                    const HNSWIndex<T, Metric>& index, char** pzErr) {
    // Create shadow tables
    int rc = create_hnsw_shadow_tables(db, schema, table, pzErr);
    if (rc != SQLITE_OK)
        return rc;

    // Begin transaction
    rc = sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("Failed to begin transaction");
        return rc;
    }

    // Save config
    std::ostringstream config_sql;
    config_sql << "INSERT OR REPLACE INTO \"" << schema << "\".\"" << table
               << "_hnsw_meta\" (key, value) VALUES ('config', ?)";

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, config_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    auto config_blob = serialize_hnsw_config<T, Metric>(index.config());
    sqlite3_bind_blob(stmt, 1, config_blob.data(), config_blob.size(), SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    // Save entry point
    std::ostringstream entry_sql;
    entry_sql << "INSERT OR REPLACE INTO \"" << schema << "\".\"" << table
              << "_hnsw_meta\" (key, value) VALUES ('entry_point', ?)";

    rc = sqlite3_prepare_v2(db, entry_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    uint64_t entry_data[2] = {index.entry_point(), index.max_layer()};
    sqlite3_bind_blob(stmt, 1, entry_data, sizeof(entry_data), SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    // Save all nodes
    std::ostringstream nodes_sql;
    nodes_sql << "INSERT OR REPLACE INTO \"" << schema << "\".\"" << table
              << "_hnsw_nodes\" (node_id, data) VALUES (?, ?)";

    rc = sqlite3_prepare_v2(db, nodes_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    for (auto it = index.begin(); it != index.end(); ++it) {
        const auto& [node_id, node] = *it;
        auto node_blob = serialize_hnsw_node(node);

        sqlite3_reset(stmt);
        sqlite3_bind_int64(stmt, 1, node_id);
        sqlite3_bind_blob(stmt, 2, node_blob.data(), node_blob.size(), SQLITE_TRANSIENT);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            if (pzErr)
                *pzErr = sqlite3_mprintf("Failed to save HNSW node %zu", node_id);
            return rc;
        }
    }

    sqlite3_finalize(stmt);

    // Commit transaction
    rc = sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    return rc;
}

/// Load HNSW index from SQLite shadow tables
template <typename T, typename Metric>
HNSWIndex<T, Metric> load_hnsw_index(sqlite3* db, const char* schema, const char* table,
                                     char** pzErr) {
    // Load config
    std::ostringstream config_sql;
    config_sql << "SELECT value FROM \"" << schema << "\".\"" << table
               << "_hnsw_meta\" WHERE key='config'";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, config_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("Failed to load HNSW config");
        throw std::runtime_error("Failed to load HNSW config");
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        if (pzErr)
            *pzErr = sqlite3_mprintf("HNSW config not found");
        throw std::runtime_error("HNSW config not found");
    }

    const void* config_blob = sqlite3_column_blob(stmt, 0);
    int config_size = sqlite3_column_bytes(stmt, 0);
    auto config = deserialize_hnsw_config<T, Metric>(config_blob, config_size);
    sqlite3_finalize(stmt);

    // Load entry point
    std::ostringstream entry_sql;
    entry_sql << "SELECT value FROM \"" << schema << "\".\"" << table
              << "_hnsw_meta\" WHERE key='entry_point'";

    rc = sqlite3_prepare_v2(db, entry_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("Failed to load HNSW entry point");
        throw std::runtime_error("Failed to load HNSW entry point");
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        if (pzErr)
            *pzErr = sqlite3_mprintf("HNSW entry point not found");
        throw std::runtime_error("HNSW entry point not found");
    }

    const uint64_t* entry_data = static_cast<const uint64_t*>(sqlite3_column_blob(stmt, 0));
    size_t entry_point_id = entry_data[0];
    size_t entry_point_layer = entry_data[1];
    sqlite3_finalize(stmt);

    // Load all nodes
    std::ostringstream nodes_sql;
    nodes_sql << "SELECT node_id, data FROM \"" << schema << "\".\"" << table
              << "_hnsw_nodes\" ORDER BY node_id";

    rc = sqlite3_prepare_v2(db, nodes_sql.str().c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("Failed to load HNSW nodes");
        throw std::runtime_error("Failed to load HNSW nodes");
    }

    std::unordered_map<size_t, HNSWNode<T>> nodes;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        size_t node_id = sqlite3_column_int64(stmt, 0);
        const void* node_blob = sqlite3_column_blob(stmt, 1);
        int node_size = sqlite3_column_bytes(stmt, 1);

        HNSWNode<T> node = deserialize_hnsw_node<T>(node_blob, node_size);
        nodes.emplace(node_id, std::move(node));
    }

    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("Failed to iterate HNSW nodes");
        throw std::runtime_error("Failed to iterate HNSW nodes");
    }

    // Use factory method to reconstruct index
    return HNSWIndex<T, Metric>::from_serialized(config, entry_point_id, entry_point_layer,
                                                 std::move(nodes));
}

} // namespace sqlite_vec_cpp::index
