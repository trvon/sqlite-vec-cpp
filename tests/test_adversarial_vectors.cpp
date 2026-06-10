// SPDX-License-Identifier: Apache-2.0 OR MIT
// Adversarial input tests: NaN/Inf/zero/denormal vectors through distance
// functions, HNSW insert/search, and quantization encode paths.
// Property under test: no crashes, no out-of-range ids, documented conventions hold.

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <sqlite-vec-cpp/distances/cosine.hpp>
#include <sqlite-vec-cpp/distances/hamming.hpp>
#include <sqlite-vec-cpp/distances/inner_product.hpp>
#include <sqlite-vec-cpp/distances/l1.hpp>
#include <sqlite-vec-cpp/distances/l2.hpp>
#include <sqlite-vec-cpp/index/hnsw.hpp>
#include <sqlite-vec-cpp/quantization/lvq.hpp>
#include <sqlite-vec-cpp/quantization/rabitq.hpp>

using namespace sqlite_vec_cpp::distances;
using namespace sqlite_vec_cpp::index;
using namespace sqlite_vec_cpp::quantization;

namespace {

constexpr size_t kDim = 32;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kDenorm = std::numeric_limits<float>::denorm_min();

std::vector<float> normal_vector(size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

std::vector<std::vector<float>> adversarial_vectors() {
    std::vector<std::vector<float>> out;

    out.push_back(std::vector<float>(kDim, 0.0f));
    out.push_back(std::vector<float>(kDim, kDenorm));
    out.push_back(std::vector<float>(kDim, kNaN));
    out.push_back(std::vector<float>(kDim, kInf));
    out.push_back(std::vector<float>(kDim, -kInf));

    auto mixed_nan = normal_vector(kDim, 11);
    mixed_nan[3] = kNaN;
    out.push_back(std::move(mixed_nan));

    auto mixed_inf = normal_vector(kDim, 12);
    mixed_inf[7] = kInf;
    mixed_inf[8] = -kInf;
    out.push_back(std::move(mixed_inf));

    auto extreme = normal_vector(kDim, 13);
    extreme[0] = std::numeric_limits<float>::max();
    extreme[1] = std::numeric_limits<float>::lowest();
    out.push_back(std::move(extreme));

    return out;
}

void test_distance_functions_no_crash() {
    std::cout << "Adversarial 1: distance functions over hostile inputs..." << std::endl;

    auto hostile = adversarial_vectors();
    auto reference = normal_vector(kDim, 1);

    size_t evaluated = 0;
    for (const auto& a : hostile) {
        for (const auto& b : hostile) {
            volatile float d1 = l2_distance(std::span<const float>(a), std::span<const float>(b));
            volatile float d2 =
                cosine_distance(std::span<const float>(a), std::span<const float>(b));
            volatile float d3 = l1_distance(std::span<const float>(a), std::span<const float>(b));
            volatile float d4 =
                inner_product_distance(std::span<const float>(a), std::span<const float>(b));
            (void)d1;
            (void)d2;
            (void)d3;
            (void)d4;
            evaluated += 4;
        }
        volatile float dr =
            l2_distance(std::span<const float>(a), std::span<const float>(reference));
        (void)dr;
        ++evaluated;
    }
    std::cout << "  ✓ " << evaluated << " evaluations survived" << std::endl;
}

void test_zero_vector_cosine_convention() {
    std::cout << "Adversarial 2: zero/denormal vector cosine convention..." << std::endl;

    std::vector<float> zero(kDim, 0.0f);
    std::vector<float> denorm(kDim, kDenorm);
    auto reference = normal_vector(kDim, 2);

    float d_zero = cosine_distance(std::span<const float>(zero), std::span<const float>(reference));
    assert(d_zero == 1.0f);

    float d_zero_zero =
        cosine_distance(std::span<const float>(zero), std::span<const float>(zero));
    assert(d_zero_zero == 1.0f);

    float d_denorm =
        cosine_distance(std::span<const float>(denorm), std::span<const float>(reference));
    assert(d_denorm == 1.0f);

    std::vector<float> small_zero(4, 0.0f);
    std::vector<float> small_ref = {0.5f, -0.5f, 0.25f, -0.25f};
    float d_small =
        cosine_distance(std::span<const float>(small_zero), std::span<const float>(small_ref));
    assert(d_small == 1.0f);

    std::cout << "  ✓ degenerate-denominator inputs return distance 1.0 in all paths"
              << std::endl;
}

void test_hamming_bitvectors() {
    std::cout << "Adversarial 3: hamming over degenerate bitvectors..." << std::endl;

    std::vector<uint8_t> zeros(16, 0x00);
    std::vector<uint8_t> ones(16, 0xFF);
    std::vector<uint8_t> single(1, 0x80);

    assert(hamming_distance(std::span<const uint8_t>(zeros), std::span<const uint8_t>(zeros),
                            128) == 0.0f);
    assert(hamming_distance(std::span<const uint8_t>(zeros), std::span<const uint8_t>(ones),
                            128) == 128.0f);
    assert(hamming_distance(std::span<const uint8_t>(single), std::span<const uint8_t>(single),
                            8) == 0.0f);

    std::cout << "  ✓ hamming conventions hold" << std::endl;
}

void test_hnsw_with_hostile_inserts() {
    std::cout << "Adversarial 4: HNSW insert/search with hostile vectors mixed in..."
              << std::endl;

    HNSWIndex<float, L2Metric<float>> index;

    std::vector<std::vector<float>> corpus;
    for (size_t i = 0; i < 200; ++i) {
        corpus.push_back(normal_vector(kDim, static_cast<uint32_t>(1000 + i)));
    }
    auto hostile = adversarial_vectors();

    size_t next_id = 0;
    for (size_t i = 0; i < corpus.size(); ++i) {
        index.insert(next_id++, std::span<const float>(corpus[i]));
        if (i % 25 == 0 && i / 25 < hostile.size()) {
            index.insert(next_id++, std::span<const float>(hostile[i / 25]));
        }
    }

    auto query = normal_vector(kDim, 5);
    auto results = index.search_read_mostly(std::span<const float>(query), 10, 100);
    assert(!results.empty());
    for (const auto& [id, dist] : results) {
        assert(id < next_id);
    }

    size_t finite_count = 0;
    for (const auto& [id, dist] : results) {
        if (std::isfinite(dist)) {
            ++finite_count;
        }
    }
    assert(finite_count >= results.size() / 2);

    for (const auto& h : hostile) {
        auto hostile_results = index.search_read_mostly(std::span<const float>(h), 5, 50);
        for (const auto& [id, dist] : hostile_results) {
            assert(id < next_id);
        }
    }

    std::cout << "  ✓ index survives hostile inserts and hostile queries" << std::endl;
}

void test_lvq_hostile_encode() {
    std::cout << "Adversarial 5: LVQ encode/decode over hostile vectors..." << std::endl;

    auto hostile = adversarial_vectors();
    for (const auto& vec : hostile) {
        auto code8 = LVQ8::encode(std::span<const float>(vec));
        assert(code8.codes.size() == vec.size());
        auto decoded = LVQ8::decode(code8);
        assert(decoded.size() == vec.size());

        auto code4 = LVQ4::encode(std::span<const float>(vec));
        assert(code4.codes.size() == (vec.size() + 1) / 2);
    }

    std::vector<float> constant(kDim, 3.25f);
    auto const_code = LVQ8::encode(std::span<const float>(constant));
    auto const_decoded = LVQ8::decode(const_code);
    for (float v : const_decoded) {
        assert(std::abs(v - 3.25f) < 0.05f);
    }

    std::cout << "  ✓ LVQ encode paths survive; constant vectors round-trip" << std::endl;
}

void test_rabitq_hostile_encode() {
    std::cout << "Adversarial 6: RaBitQ encode over hostile vectors..." << std::endl;

    auto hostile = adversarial_vectors();
    RaBitQ rq(std::vector<float>(kDim, 0.0f));

    for (const auto& vec : hostile) {
        auto code = rq.encode(std::span<const float>(vec));
        assert(code.bits.size() == (kDim + 7) / 8);
    }

    std::cout << "  ✓ RaBitQ encode survives hostile inputs" << std::endl;
}

} // namespace

int main() {
    std::cout << "=== Adversarial Vector Tests ===" << std::endl;

    test_distance_functions_no_crash();
    test_zero_vector_cosine_convention();
    test_hamming_bitvectors();
    test_hnsw_with_hostile_inserts();
    test_lvq_hostile_encode();
    test_rabitq_hostile_encode();

    std::cout << "=== All adversarial vector tests passed ===" << std::endl;
    return 0;
}
