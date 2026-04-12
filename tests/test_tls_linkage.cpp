#include <cassert>

#include <sqlite-vec-cpp/index/hnsw_threading.hpp>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

int exercise_tls_linkage_helper();

int main() {
    using namespace sqlite_vec_cpp::index;

    ThreadLocalRNG rng(7);
    const auto node = static_cast<size_t>(rng.random_uint(128));

    auto& visited = ThreadLocalVisitedPool::get(128);
    const bool first_visit = visited.visit(node);
    assert(first_visit);
    assert(visited.is_visited(node));

    assert(exercise_tls_linkage_helper() == 1);
    return 0;
}
