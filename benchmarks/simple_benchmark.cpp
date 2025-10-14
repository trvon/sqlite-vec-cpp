// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright 2025 - Present, Trevon Hanna
// Simple benchmark without external dependencies

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "vec_core.hpp"

using namespace sqlite_vec;
using namespace std::chrono;

template <typename T>
std::vector<T> generate_random_vector(size_t dim, T min = T{0}, T max = T{1}) {
    std::mt19937 gen(42); // Fixed seed

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min, max);
        std::vector<T> vec(dim);
        std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
        return vec;
    } else {
        std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));
        std::vector<T> vec(dim);
        std::generate(vec.begin(), vec.end(), [&]() { return static_cast<T>(dist(gen)); });
        return vec;
    }
}

template <typename Func> double benchmark_function(Func&& func, int iterations) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    return static_cast<double>(duration) / iterations;
}

void print_header() {
    std::cout << "\n============================================================\n";
    std::cout << "           sqlite-vec-cpp Distance Benchmark\n";
    std::cout << "============================================================\n\n";
    std::cout << std::left << std::setw(30) << "Test" << std::right << std::setw(15)
              << "Avg Time (μs)" << std::setw(15) << "Throughput" << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void print_result(const std::string& name, double time_us, size_t elements) {
    double throughput = (elements / time_us) * 1000.0; // Elements per millisecond
    std::cout << std::left << std::setw(30) << name << std::right << std::fixed
              << std::setprecision(3) << std::setw(15) << time_us << std::setw(12)
              << static_cast<int>(throughput) << " K/ms" << "\n";
}

int main() {
    const int ITERATIONS = 10000;

    print_header();

    // ========================================================================
    // L2 Distance Benchmarks
    // ========================================================================
    {
        auto a = generate_random_vector<float>(128);
        auto b = generate_random_vector<float>(128);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L2 (float, dim=128)", time, 128);
    }

    {
        auto a = generate_random_vector<float>(256);
        auto b = generate_random_vector<float>(256);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L2 (float, dim=256)", time, 256);
    }

    {
        auto a = generate_random_vector<float>(512);
        auto b = generate_random_vector<float>(512);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L2 (float, dim=512)", time, 512);
    }

    {
        auto a = generate_random_vector<float>(1536);
        auto b = generate_random_vector<float>(1536);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l2_sqeuclidean(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L2 (float, dim=1536)", time, 1536);
    }

    {
        auto a = generate_random_vector<int8_t>(128, int8_t{-128}, int8_t{127});
        auto b = generate_random_vector<int8_t>(128, int8_t{-128}, int8_t{127});
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l2_sqeuclidean(std::span<const int8_t>(a), std::span<const int8_t>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L2 (int8, dim=128)", time, 128);
    }

    // ========================================================================
    // L1 Distance Benchmarks
    // ========================================================================
    {
        auto a = generate_random_vector<float>(128);
        auto b = generate_random_vector<float>(128);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l1(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L1 (float, dim=128)", time, 128);
    }

    {
        auto a = generate_random_vector<float>(1536);
        auto b = generate_random_vector<float>(1536);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_l1(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("L1 (float, dim=1536)", time, 1536);
    }

    // ========================================================================
    // Cosine Distance Benchmarks
    // ========================================================================
    {
        auto a = generate_random_vector<float>(128);
        auto b = generate_random_vector<float>(128);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_cosine(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("Cosine (float, dim=128)", time, 128);
    }

    {
        auto a = generate_random_vector<float>(1536);
        auto b = generate_random_vector<float>(1536);
        double time = benchmark_function(
            [&]() {
                volatile float result =
                    distance_cosine(std::span<const float>(a), std::span<const float>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("Cosine (float, dim=1536)", time, 1536);
    }

    // ========================================================================
    // Hamming Distance Benchmarks
    // ========================================================================
    {
        auto a = generate_random_vector<unsigned char>(128, (unsigned char)0, (unsigned char)255);
        auto b = generate_random_vector<unsigned char>(128, (unsigned char)0, (unsigned char)255);
        double time = benchmark_function(
            [&]() {
                volatile int result = distance_hamming(std::span<const unsigned char>(a),
                                                       std::span<const unsigned char>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("Hamming (dim=128)", time, 128);
    }

    {
        auto a = generate_random_vector<unsigned char>(1536, (unsigned char)0, (unsigned char)255);
        auto b = generate_random_vector<unsigned char>(1536, (unsigned char)0, (unsigned char)255);
        double time = benchmark_function(
            [&]() {
                volatile int result = distance_hamming(std::span<const unsigned char>(a),
                                                       std::span<const unsigned char>(b));
                (void)result;
            },
            ITERATIONS);
        print_result("Hamming (dim=1536)", time, 1536);
    }

    // ========================================================================
    // Batch operation simulation
    // ========================================================================
    {
        std::vector<std::vector<float>> queries;
        for (int i = 0; i < 100; ++i) {
            queries.push_back(generate_random_vector<float>(128));
        }
        auto target = generate_random_vector<float>(128);

        double time = benchmark_function(
            [&]() {
                for (const auto& query : queries) {
                    volatile float result = distance_l2_sqeuclidean(std::span<const float>(query),
                                                                    std::span<const float>(target));
                    (void)result;
                }
            },
            100);
        print_result("Batch L2 (100x128)", time, 12800);
    }

    std::cout << "\n============================================================\n";
    std::cout << "Benchmark completed. Each test ran " << ITERATIONS << " iterations.\n";
    std::cout << "============================================================\n\n";

    // SIMD status
    std::cout << "SIMD Status:\n";
#ifdef SQLITE_VEC_ENABLE_AVX
    std::cout << "  - AVX: ENABLED\n";
#else
    std::cout << "  - AVX: disabled\n";
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
    std::cout << "  - NEON: ENABLED\n";
#else
    std::cout << "  - NEON: disabled\n";
#endif

    return 0;
}
