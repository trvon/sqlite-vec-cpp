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
    assert(!visited.visit(node));

    auto& reset_visited = ThreadLocalVisitedPool::get(128);
    assert(&reset_visited == &visited);
    assert(!reset_visited.is_visited(node));
    assert(reset_visited.visit(node));

    auto& grown_visited = ThreadLocalVisitedPool::get(8192);
    assert(&grown_visited == &visited);
    assert(grown_visited.capacity() >= 8192);
    assert(!grown_visited.is_visited(node));

    assert(exercise_tls_linkage_helper() == 1);
    return 0;
}
