// SPDX-License-Identifier: Apache-2.0 OR MIT
// Precision/accuracy benchmarks for distance metrics

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <vector>
#include <benchmark/benchmark.h>
#include <sqlite-vec-cpp/distances/cosine.hpp>
#include <sqlite-vec-cpp/distances/inner_product.hpp>
#include <sqlite-vec-cpp/distances/l2.hpp>

using namespace sqlite_vec_cpp::distances;

struct ErrorStats {
    double mean_abs = 0.0;
    double mean_rel = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double p95_abs = 0.0;
    double p95_rel = 0.0;
};

static std::vector<float> generate_vector(size_t dim, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

static double l2_distance_double(std::span<const float> a, std::span<const float> b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static double cosine_distance_double(std::span<const float> a, std::span<const float> b) {
    double dot = 0.0;
    double a_mag = 0.0;
    double b_mag = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        dot += av * bv;
        a_mag += av * av;
        b_mag += bv * bv;
    }
    double denom = std::sqrt(a_mag) * std::sqrt(b_mag);
    if (denom < 1e-12) {
        return 1.0;
    }
    return 1.0 - (dot / denom);
}

static double inner_product_distance_double(std::span<const float> a, std::span<const float> b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return 1.0 - dot;
}

template <typename MetricFn, typename BaselineFn>
static ErrorStats compute_error_stats(size_t dim, size_t samples, float scale, MetricFn metric,
                                      BaselineFn baseline) {
    std::mt19937 rng(42);
    std::vector<double> abs_errors;
    std::vector<double> rel_errors;
    abs_errors.reserve(samples);
    rel_errors.reserve(samples);

    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;

    for (size_t i = 0; i < samples; ++i) {
        auto a = generate_vector(dim, rng, scale);
        auto b = generate_vector(dim, rng, scale);

        float approx = metric(std::span<const float>(a), std::span<const float>(b));
        double exact = baseline(std::span<const float>(a), std::span<const float>(b));

        double abs_err = std::abs(static_cast<double>(approx) - exact);
        double rel_err = 0.0;
        double denom = std::abs(exact);
        if (denom > 1e-12) {
            rel_err = abs_err / denom;
        }

        abs_errors.push_back(abs_err);
        rel_errors.push_back(rel_err);
        sum_abs += abs_err;
        sum_rel += rel_err;
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }

    std::sort(abs_errors.begin(), abs_errors.end());
    std::sort(rel_errors.begin(), rel_errors.end());
    size_t p95 = static_cast<size_t>(0.95 * static_cast<double>(samples - 1));

    ErrorStats stats;
    stats.mean_abs = sum_abs / static_cast<double>(samples);
    stats.mean_rel = sum_rel / static_cast<double>(samples);
    stats.max_abs = max_abs;
    stats.max_rel = max_rel;
    stats.p95_abs = abs_errors[p95];
    stats.p95_rel = rel_errors[p95];
    return stats;
}

static void apply_counters(benchmark::State& state, const ErrorStats& stats) {
    state.counters["mean_abs"] = stats.mean_abs;
    state.counters["mean_rel"] = stats.mean_rel;
    state.counters["max_abs"] = stats.max_abs;
    state.counters["max_rel"] = stats.max_rel;
    state.counters["p95_abs"] = stats.p95_abs;
    state.counters["p95_rel"] = stats.p95_rel;
}

static void BM_L2_Precision(benchmark::State& state) {
    size_t dim = static_cast<size_t>(state.range(0));
    float scale = static_cast<float>(state.range(1));
    constexpr size_t samples = 1000;

    ErrorStats stats;
    for (auto _ : state) {
        state.PauseTiming();
        stats = compute_error_stats(dim, samples, scale, L2Metric<float>{},
                                    [](std::span<const float> a, std::span<const float> b) {
                                        return l2_distance_double(a, b);
                                    });
        state.ResumeTiming();
        benchmark::DoNotOptimize(stats.mean_abs);
    }
    apply_counters(state, stats);
}

static void BM_Cosine_Precision(benchmark::State& state) {
    size_t dim = static_cast<size_t>(state.range(0));
    float scale = static_cast<float>(state.range(1));
    constexpr size_t samples = 1000;

    ErrorStats stats;
    for (auto _ : state) {
        state.PauseTiming();
        stats = compute_error_stats(dim, samples, scale, CosineMetric<float>{},
                                    [](std::span<const float> a, std::span<const float> b) {
                                        return cosine_distance_double(a, b);
                                    });
        state.ResumeTiming();
        benchmark::DoNotOptimize(stats.mean_abs);
    }
    apply_counters(state, stats);
}

static void BM_InnerProduct_Precision(benchmark::State& state) {
    size_t dim = static_cast<size_t>(state.range(0));
    float scale = static_cast<float>(state.range(1));
    constexpr size_t samples = 1000;

    ErrorStats stats;
    for (auto _ : state) {
        state.PauseTiming();
        stats = compute_error_stats(dim, samples, scale, InnerProductMetric<float>{},
                                    [](std::span<const float> a, std::span<const float> b) {
                                        return inner_product_distance_double(a, b);
                                    });
        state.ResumeTiming();
        benchmark::DoNotOptimize(stats.mean_abs);
    }
    apply_counters(state, stats);
}

BENCHMARK(BM_L2_Precision)->Args({384, 1})->Args({384, 1000})->Args({384, 1000000});
BENCHMARK(BM_Cosine_Precision)->Args({384, 1})->Args({384, 1000})->Args({384, 1000000});
BENCHMARK(BM_InnerProduct_Precision)->Args({384, 1})->Args({384, 1000})->Args({384, 1000000});

BENCHMARK_MAIN();
