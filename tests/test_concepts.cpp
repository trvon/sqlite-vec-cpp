#include <iostream>
#include <vector>
#include <sqlite-vec-cpp/concepts/distance_metric.hpp>
#include <sqlite-vec-cpp/concepts/vector_element.hpp>
#include <sqlite-vec-cpp/utils/error.hpp>
#include <sqlite-vec-cpp/vector_view.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::concepts;

// Simple test metric to verify concept checking
template <VectorElement T> struct TestMetric {
    float operator()(std::span<const T> a, std::span<const T> b) const {
        return 0.0f; // Dummy implementation
    }
};

int main() {
    // Test VectorElement concept
    static_assert(VectorElement<float>, "float should satisfy VectorElement");
    static_assert(VectorElement<std::int8_t>, "int8_t should satisfy VectorElement");
    static_assert(!VectorElement<bool>, "bool should NOT satisfy VectorElement");

    // Test FloatingPointElement concept
    static_assert(FloatingPointElement<float>, "float should satisfy FloatingPointElement");
    static_assert(!FloatingPointElement<int>, "int should NOT satisfy FloatingPointElement");

    // Test DistanceMetric concept
    static_assert(DistanceMetric<TestMetric<float>, float>,
                  "TestMetric should satisfy DistanceMetric");

    // Test VectorView
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    VectorView<float> view(data);

    if (view.size() != 4) {
        std::cerr << "VectorView size mismatch\n";
        return 1;
    }

    if (view[0] != 1.0f || view[3] != 4.0f) {
        std::cerr << "VectorView element access failed\n";
        return 1;
    }

    // Test blob conversion
    auto blob = view.to_blob();
    if (blob.size() != 4 * sizeof(float)) {
        std::cerr << "Blob size mismatch\n";
        return 1;
    }

    // Test vector_from_blob
    auto restored = vector_from_blob<float>(blob.data(), blob.size());
    if (restored.size() != 4 || restored[0] != 1.0f) {
        std::cerr << "Blob deserialization failed\n";
        return 1;
    }

    // Test error handling
    VoidResult success = ok();
    if (!success) {
        std::cerr << "Expected success result\n";
        return 1;
    }

    VoidResult failure = err_void(Error::invalid_argument("test error"));
    if (failure) {
        std::cerr << "Expected failure result\n";
        return 1;
    }

    if (failure.error().code != ErrorCode::InvalidArgument) {
        std::cerr << "Error code mismatch\n";
        return 1;
    }

    std::cout << "All concept tests passed!\n";
    return 0;
}
