// Benchmark: HNSW recall and latency under sustained insert/delete churn.
//
// Quantifies the delete-policy tradeoff before any repair-on-delete work
// (MN-RU, arXiv:2407.07871) is considered:
//   - soft:    remove() only — deleted nodes stay as traversal waypoints
//   - isolate: remove() + isolate_deleted() each cycle — edges to deleted
//              nodes are cut, which can strand their former in-neighbors
//   - compact: remove() + full compact() rebuild each cycle (upper bound)
//
// Each cycle deletes `batch` random active ids and inserts `batch` new
// vectors. After each cycle we measure recall@k against brute force over the
// active set, plus query latency. A fresh-built index over the same active
// set is measured at the final cycle as the reference ceiling.
//
// Usage:
//   ./hnsw_churn_benchmark [--corpus N] [--dim D] [--cycles C] [--batch B]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

namespace {

using Index = HNSWIndex<float, L2Metric<float>>;

std::vector<float> generate_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec)
        v = dist(rng);
    return vec;
}

struct ActiveSet {
    std::vector<std::vector<float>> vectors; // indexed by id
    std::vector<size_t> active_ids;
};

std::vector<size_t> brute_force_knn(std::span<const float> query, const ActiveSet& set, size_t k) {
    L2Metric<float> metric;
    std::vector<std::pair<size_t, float>> distances;
    distances.reserve(set.active_ids.size());
    for (size_t id : set.active_ids) {
        distances.emplace_back(id, metric(query, std::span<const float>(set.vectors[id])));
    }
    std::partial_sort(distances.begin(), distances.begin() + std::min(k, distances.size()),
                      distances.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
    distances.resize(std::min(k, distances.size()));
    std::vector<size_t> ids;
    ids.reserve(distances.size());
    for (const auto& [id, d] : distances)
        ids.push_back(id);
    return ids;
}

struct CycleStats {
    float recall = 0.0f;
    double avg_latency_us = 0.0;
    size_t graph_nodes = 0;
    size_t deleted = 0;
};

CycleStats measure(const Index& index, const ActiveSet& set,
                   const std::vector<std::vector<float>>& queries, size_t k, size_t ef_search) {
    CycleStats stats;
    stats.graph_nodes = index.size();
    stats.deleted = index.deleted_count();

    size_t hits = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> all_results;
    all_results.reserve(queries.size());
    for (const auto& q : queries) {
        auto results = index.search_read_mostly(std::span<const float>(q), k, ef_search);
        std::vector<size_t> ids;
        ids.reserve(results.size());
        for (const auto& [id, d] : results)
            ids.push_back(id);
        all_results.push_back(std::move(ids));
    }
    auto end = std::chrono::high_resolution_clock::now();
    stats.avg_latency_us = std::chrono::duration<double, std::micro>(end - start).count() /
                           static_cast<double>(queries.size());

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto truth = brute_force_knn(std::span<const float>(queries[qi]), set, k);
        std::unordered_set<size_t> truth_set(truth.begin(), truth.end());
        for (size_t id : all_results[qi]) {
            hits += truth_set.contains(id) ? 1 : 0;
        }
    }
    stats.recall = static_cast<float>(hits) /
                   static_cast<float>(queries.size() * std::min(k, set.active_ids.size()));
    return stats;
}

enum class Policy { Soft, Isolate, Compact };

const char* policy_name(Policy p) {
    switch (p) {
        case Policy::Soft:
            return "soft";
        case Policy::Isolate:
            return "isolate";
        case Policy::Compact:
            return "compact";
    }
    return "?";
}

} // namespace

int main(int argc, char* argv[]) {
    size_t corpus_size = 10000;
    size_t dim = 128;
    size_t cycles = 10;
    size_t batch = 500;
    size_t num_queries = 50;
    size_t k = 10;
    size_t ef_search = 100;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_size = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            dim = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--cycles") == 0 && i + 1 < argc) {
            cycles = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch = std::stoul(argv[++i]);
        }
    }

    std::cout << "=== HNSW Churn Benchmark ===" << std::endl;
    std::cout << "corpus=" << corpus_size << " dim=" << dim << " cycles=" << cycles
              << " batch=" << batch << " k=" << k << " ef_search=" << ef_search << std::endl;

    for (Policy policy : {Policy::Soft, Policy::Isolate, Policy::Compact}) {
        std::mt19937 rng(42);
        std::mt19937 churn_rng(777);

        ActiveSet set;
        set.vectors.reserve(corpus_size + cycles * batch);
        for (size_t i = 0; i < corpus_size; ++i) {
            set.vectors.push_back(generate_vector(dim, rng));
            set.active_ids.push_back(i);
        }
        std::vector<std::vector<float>> queries;
        for (size_t i = 0; i < num_queries; ++i) {
            queries.push_back(generate_vector(dim, rng));
        }

        Index index;
        for (size_t i = 0; i < corpus_size; ++i) {
            index.insert_single_threaded(i, std::span<const float>(set.vectors[i]));
        }

        std::cout << std::endl
                  << "--- policy=" << policy_name(policy) << " ---" << std::endl;
        std::cout << "| Cycle | Recall@10 | Latency (us) | Nodes | Deleted | Churn (ms) |"
                  << std::endl;
        std::cout << "|-------|-----------|--------------|-------|---------|------------|"
                  << std::endl;

        auto baseline = measure(index, set, queries, k, ef_search);
        std::cout << "| 0 | " << std::fixed << std::setprecision(1) << baseline.recall * 100
                  << "% | " << baseline.avg_latency_us << " | " << baseline.graph_nodes << " | "
                  << baseline.deleted << " | - |" << std::endl;

        size_t next_id = corpus_size;
        std::optional<Index> compacted_holder;
        Index* live = &index;

        for (size_t cycle = 1; cycle <= cycles; ++cycle) {
            auto churn_start = std::chrono::high_resolution_clock::now();

            std::shuffle(set.active_ids.begin(), set.active_ids.end(), churn_rng);
            for (size_t b = 0; b < batch && !set.active_ids.empty(); ++b) {
                live->remove(set.active_ids.back());
                set.active_ids.pop_back();
            }
            for (size_t b = 0; b < batch; ++b) {
                set.vectors.push_back(generate_vector(dim, churn_rng));
                set.active_ids.push_back(next_id);
                live->insert(next_id, std::span<const float>(set.vectors[next_id]));
                ++next_id;
            }

            if (policy == Policy::Isolate) {
                live->isolate_deleted();
            } else if (policy == Policy::Compact) {
                compacted_holder = live->compact();
                live = &*compacted_holder;
            }

            auto churn_end = std::chrono::high_resolution_clock::now();
            double churn_ms =
                std::chrono::duration<double, std::milli>(churn_end - churn_start).count();

            auto stats = measure(*live, set, queries, k, ef_search);
            std::cout << "| " << cycle << " | " << std::fixed << std::setprecision(1)
                      << stats.recall * 100 << "% | " << stats.avg_latency_us << " | "
                      << stats.graph_nodes << " | " << stats.deleted << " | "
                      << std::setprecision(1) << churn_ms << " |" << std::endl;
        }

        Index fresh;
        for (size_t id : set.active_ids) {
            fresh.insert_single_threaded(id, std::span<const float>(set.vectors[id]));
        }
        auto fresh_stats = measure(fresh, set, queries, k, ef_search);
        std::cout << "| fresh | " << std::fixed << std::setprecision(1)
                  << fresh_stats.recall * 100 << "% | " << fresh_stats.avg_latency_us << " | "
                  << fresh_stats.graph_nodes << " | 0 | - |" << std::endl;
    }

    return 0;
}
