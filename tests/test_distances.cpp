#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <sqlite-vec-cpp/sqlite_vec.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::distances;

// Helper to compare floats with tolerance
bool approx_equal(float a, float b, float epsilon = 0.0001f) {
    return std::abs(a - b) < epsilon;
}

void test_l2_distance() {
    std::cout << "Testing L2 distance..." << std::endl;

    // Test with float vectors
    std::vector<float> a_float = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_float = {2.0f, 3.0f, 4.0f, 5.0f};

    VectorView<const float> view_a(a_float);
    VectorView<const float> view_b(b_float);

    float dist = l2_distance(view_a.span(), view_b.span());
    // Expected: sqrt((1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2) = sqrt(4) = 2.0
    assert(approx_equal(dist, 2.0f));

    // Test with L2Metric functor
    L2Metric<float> metric;
    float dist2 = metric(view_a.span(), view_b.span());
    assert(approx_equal(dist2, 2.0f));

    // Test with int8_t vectors
    std::vector<std::int8_t> a_int8 = {1, 2, 3, 4};
    std::vector<std::int8_t> b_int8 = {2, 3, 4, 5};

    VectorView<const std::int8_t> view_a_int8(a_int8);
    VectorView<const std::int8_t> view_b_int8(b_int8);

    float dist_int8 = l2_distance(view_a_int8.span(), view_b_int8.span());
    assert(approx_equal(dist_int8, 2.0f));

    // Test zero distance (same vector)
    float dist_zero = l2_distance(view_a.span(), view_a.span());
    assert(approx_equal(dist_zero, 0.0f));

    std::cout << "  L2 distance tests passed!" << std::endl;
}

void test_l1_distance() {
    std::cout << "Testing L1 distance..." << std::endl;

    // Test with float vectors
    std::vector<float> a_float = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_float = {2.0f, 3.0f, 4.0f, 5.0f};

    VectorView<const float> view_a(a_float);
    VectorView<const float> view_b(b_float);

    float dist = l1_distance(view_a.span(), view_b.span());
    // Expected: |1-2| + |2-3| + |3-4| + |4-5| = 1+1+1+1 = 4.0
    assert(approx_equal(dist, 4.0f));

    // Test with L1Metric functor
    L1Metric<float> metric;
    float dist2 = metric(view_a.span(), view_b.span());
    assert(approx_equal(dist2, 4.0f));

    // Test with int8_t vectors
    std::vector<std::int8_t> a_int8 = {1, 2, 3, 4};
    std::vector<std::int8_t> b_int8 = {2, 3, 4, 5};

    VectorView<const std::int8_t> view_a_int8(a_int8);
    VectorView<const std::int8_t> view_b_int8(b_int8);

    float dist_int8 = l1_distance(view_a_int8.span(), view_b_int8.span());
    assert(approx_equal(dist_int8, 4.0f));

    // Test zero distance
    float dist_zero = l1_distance(view_a.span(), view_a.span());
    assert(approx_equal(dist_zero, 0.0f));

    std::cout << "  L1 distance tests passed!" << std::endl;
}

void test_cosine_distance() {
    std::cout << "Testing cosine distance..." << std::endl;

    // Test with orthogonal vectors (90 degrees)
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};

    VectorView<const float> view_a(a);
    VectorView<const float> view_b(b);

    float dist = cosine_distance(view_a.span(), view_b.span());
    // Expected: 1 - cos(90°) = 1 - 0 = 1.0
    assert(approx_equal(dist, 1.0f));

    // Test with parallel vectors (0 degrees)
    std::vector<float> c = {1.0f, 2.0f, 3.0f};
    std::vector<float> d = {2.0f, 4.0f, 6.0f}; // c * 2

    VectorView<const float> view_c(c);
    VectorView<const float> view_d(d);

    float dist_parallel = cosine_distance(view_c.span(), view_d.span());
    // Expected: 1 - cos(0°) = 1 - 1 = 0.0
    assert(approx_equal(dist_parallel, 0.0f));

    // Test with CosineMetric functor
    CosineMetric<float> metric;
    float dist2 = metric(view_a.span(), view_b.span());
    assert(approx_equal(dist2, 1.0f));

    // Test with int8_t
    std::vector<std::int8_t> a_int8 = {1, 0, 0};
    std::vector<std::int8_t> b_int8 = {0, 1, 0};

    VectorView<const std::int8_t> view_a_int8(a_int8);
    VectorView<const std::int8_t> view_b_int8(b_int8);

    float dist_int8 = cosine_distance(view_a_int8.span(), view_b_int8.span());
    assert(approx_equal(dist_int8, 1.0f));

    std::cout << "  Cosine distance tests passed!" << std::endl;
}

void test_hamming_distance() {
    std::cout << "Testing hamming distance..." << std::endl;

    // Test with bitvectors (represented as uint8_t arrays)
    std::vector<std::uint8_t> a = {0b10101010, 0b11110000}; // 16 bits total
    std::vector<std::uint8_t> b = {0b10101011, 0b11110001}; // Differs in 2 bits

    VectorView<const std::uint8_t> view_a(a);
    VectorView<const std::uint8_t> view_b(b);

    float dist = hamming_distance(view_a.span(), view_b.span(), 16);
    // Expected: 2.0 (bit 0 and bit 8 differ)
    assert(approx_equal(dist, 2.0f));

    // Test with HammingMetric functor
    HammingMetric metric(16);
    float dist2 = metric(view_a.span(), view_b.span());
    assert(approx_equal(dist2, 2.0f));

    // Test all zeros vs all ones
    std::vector<std::uint8_t> zeros = {0x00};
    std::vector<std::uint8_t> ones = {0xFF};

    VectorView<const std::uint8_t> view_zeros(zeros);
    VectorView<const std::uint8_t> view_ones(ones);

    float dist_all_diff = hamming_distance(view_zeros.span(), view_ones.span(), 8);
    assert(approx_equal(dist_all_diff, 8.0f)); // All 8 bits differ

    // Test same vectors
    float dist_zero = hamming_distance(view_a.span(), view_a.span(), 16);
    assert(approx_equal(dist_zero, 0.0f));

    std::cout << "  Hamming distance tests passed!" << std::endl;
}

void test_metric_traits() {
    std::cout << "Testing metric traits..." << std::endl;

    using namespace concepts::traits;

    // L1 and L2 are proper metrics (symmetric, triangle inequality)
    static_assert(is_symmetric_v<L1Metric<float>>);
    static_assert(is_metric_space_v<L1Metric<float>>);
    static_assert(is_symmetric_v<L2Metric<float>>);
    static_assert(is_metric_space_v<L2Metric<float>>);

    // Cosine is symmetric but NOT a proper metric (no triangle inequality)
    static_assert(is_symmetric_v<CosineMetric<float>>);
    static_assert(!is_metric_space_v<CosineMetric<float>>);
    static_assert(is_similarity_v<CosineMetric<float>>);

    // Hamming is a proper metric
    static_assert(is_symmetric_v<HammingMetric>);
    static_assert(is_metric_space_v<HammingMetric>);

    std::cout << "  Metric traits tests passed!" << std::endl;
}

int main() {
    try {
        test_l2_distance();
        test_l1_distance();
        test_cosine_distance();
        test_hamming_distance();
        test_metric_traits();

        std::cout << "\nAll distance metric tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
