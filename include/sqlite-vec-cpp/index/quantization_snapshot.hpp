#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sqlite_vec_cpp::index {

/// Point-in-time snapshot of index vectors for quantization building.
/// Captured under the index read lock so stores never iterate a live graph.
struct QuantizationSnapshot {
    struct Entry {
        size_t dense_id;
        std::vector<float> vector; // always float32, pre-converted from StorageT
    };
    std::vector<Entry> entries;
    size_t dim = 0;
    uint64_t generation = 0; // mutation_generation_ at capture time
};

} // namespace sqlite_vec_cpp::index
