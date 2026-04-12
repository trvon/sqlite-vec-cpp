#include <sqlite-vec-cpp/index/hnsw_threading.hpp>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

int exercise_tls_linkage_helper() {
    using namespace sqlite_vec_cpp::index;

    ThreadLocalRNG rng(11);
    const auto node = rng.random_layer(1.0f) % 256;

    auto& visited = ThreadLocalVisitedPool::get(256);
    visited.visit(node);
    return visited.is_visited(node) ? 1 : 0;
}
