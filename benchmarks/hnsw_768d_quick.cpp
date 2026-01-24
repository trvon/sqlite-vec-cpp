// Quick benchmark: M=24 vs M=32 for 768d (latency only, no ground truth)
#include <chrono>
#include <cstdio>
#include <random>
#include <span>
#include <vector>
#include <sqlite-vec-cpp/distances/cosine.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>

using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::distances;

std::vector<float> gen_vec(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (auto& v : vec) {
        v = dist(rng);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) {
        v /= norm;
    }
    return vec;
}

int main() {
    constexpr size_t corpus = 10000, dim = 768, k = 10, queries = 100;
    std::mt19937 rng(42);

    printf("768d Benchmark: M=24 vs M=32 (10K corpus, 100 queries)\n\n");

    // Generate data once
    std::vector<std::vector<float>> vecs, qvecs;
    for (size_t i = 0; i < corpus; ++i)
        vecs.push_back(gen_vec(dim, rng));
    for (size_t i = 0; i < queries; ++i)
        qvecs.push_back(gen_vec(dim, rng));

    for (auto [M, M_max, M_max_0] : {std::tuple{24, 48, 96}, std::tuple{32, 64, 128}}) {
        HNSWIndex<float, CosineMetric<float>>::Config cfg;
        cfg.M = M;
        cfg.M_max = M_max;
        cfg.M_max_0 = M_max_0;
        cfg.ef_construction = 200;
        HNSWIndex<float, CosineMetric<float>> idx(cfg);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < corpus; ++i)
            idx.insert(i, std::span<const float>{vecs[i]});
        auto t2 = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        printf("M=%d: build=%.0fms", M, build_ms);

        for (size_t ef : {50, 100, 200}) {
            auto s = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < queries; ++q)
                idx.search(std::span<const float>{qvecs[q]}, k, ef);
            auto e = std::chrono::high_resolution_clock::now();
            double lat_us = std::chrono::duration<double, std::micro>(e - s).count() / queries;
            printf(" | ef=%zu: %.0fus", ef, lat_us);
        }
        printf("\n");
    }
    return 0;
}
