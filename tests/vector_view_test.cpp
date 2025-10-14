// SPDX-License-Identifier: MIT
// Copyright (c) 2025 YAMS Contributors
// Tests for VectorView wrapper and type-safe span operations

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../include/sqlite-vec-cpp/concepts.hpp"
#include "../include/sqlite-vec-cpp/vector_view.hpp"

#include <array>
#include <vector>

using namespace sqlite_vec_cpp;

TEST_CASE("VectorView: Basic construction", "[vector_view]") {
    SECTION("From std::vector") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        VectorView<float> view(data);

        REQUIRE(view.size() == 4);
        REQUIRE(view.dimensions() == 4);
        REQUIRE(!view.empty());
    }

    SECTION("From std::array") {
        std::array<double, 3> data = {1.0, 2.0, 3.0};
        VectorView<double> view(data);

        REQUIRE(view.size() == 3);
        REQUIRE(view.dimensions() == 3);
    }

    SECTION("From std::span") {
        std::vector<int32_t> data = {10, 20, 30};
        std::span<const int32_t> sp{data};
        VectorView<int32_t> view(sp);

        REQUIRE(view.size() == 3);
    }

    SECTION("From pointer + size") {
        float data[] = {1.0f, 2.0f, 3.0f};
        VectorView<float> view(data, 3);

        REQUIRE(view.size() == 3);
        REQUIRE(view[0] == 1.0f);
        REQUIRE(view[2] == 3.0f);
    }
}

TEST_CASE("VectorView: Element access", "[vector_view]") {
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f};
    VectorView<float> view(data);

    SECTION("Subscript operator") {
        REQUIRE(view[0] == 10.0f);
        REQUIRE(view[1] == 20.0f);
        REQUIRE(view[2] == 30.0f);
        REQUIRE(view[3] == 40.0f);
    }

    SECTION("front() and back()") {
        REQUIRE(view.front() == 10.0f);
        REQUIRE(view.back() == 40.0f);
    }

    SECTION("data() pointer") {
        REQUIRE(view.data() == data.data());
    }
}

TEST_CASE("VectorView: Iteration", "[vector_view]") {
    std::vector<int8_t> data = {1, 2, 3, 4, 5};
    VectorView<int8_t> view(data);

    SECTION("Range-based for loop") {
        int sum = std::accumulate(view.begin(), view.end(), 0);
        REQUIRE(sum == 15);
    }

    SECTION("Iterator arithmetic") {
        auto it = view.begin();
        REQUIRE(*it == 1);
        ++it;
        REQUIRE(*it == 2);
        std::advance(it, 2);
        REQUIRE(*it == 4);
    }

    SECTION("Distance") {
        REQUIRE(std::distance(view.begin(), view.end()) == 5);
    }
}

TEST_CASE("VectorView: Subviews", "[vector_view]") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    VectorView<float> view(data);

    SECTION("subspan") {
        auto sub = view.subspan(1, 3);
        REQUIRE(sub.size() == 3);
        REQUIRE(sub[0] == 2.0f);
        REQUIRE(sub[1] == 3.0f);
        REQUIRE(sub[2] == 4.0f);
    }

    SECTION("first") {
        auto first = view.first(2);
        REQUIRE(first.size() == 2);
        REQUIRE(first[0] == 1.0f);
        REQUIRE(first[1] == 2.0f);
    }

    SECTION("last") {
        auto last = view.last(2);
        REQUIRE(last.size() == 2);
        REQUIRE(last[0] == 4.0f);
        REQUIRE(last[1] == 5.0f);
    }
}

TEST_CASE("VectorView: Byte representation", "[vector_view]") {
    std::vector<float> data = {1.0f, 2.0f};
    VectorView<float> view(data);

    SECTION("size_bytes") {
        REQUIRE(view.size_bytes() == 2 * sizeof(float));
    }

    SECTION("as_bytes") {
        auto bytes = view.as_bytes();
        REQUIRE(bytes.size() == 2 * sizeof(float));
    }

    SECTION("to_blob") {
        auto blob = view.to_blob();
        REQUIRE(blob.size() == 2 * sizeof(float));

        // Verify data integrity
        float reconstructed[2];
        std::memcpy(reconstructed, blob.data(), blob.size());
        REQUIRE(reconstructed[0] == 1.0f);
        REQUIRE(reconstructed[1] == 2.0f);
    }
}

TEST_CASE("VectorView: Empty and zero-sized", "[vector_view]") {
    SECTION("Empty vector") {
        std::vector<float> data;
        VectorView<float> view(data);

        REQUIRE(view.empty());
        REQUIRE(view.size() == 0);
        REQUIRE(view.dimensions() == 0);
    }

    SECTION("Null pointer with zero size") {
        VectorView<float> view(nullptr, 0);

        REQUIRE(view.empty());
        REQUIRE(view.size() == 0);
    }
}

TEST_CASE("VectorView: Type conversions", "[vector_view]") {
    SECTION("From int8_t vector") {
        std::vector<int8_t> data = {-128, 0, 127};
        VectorView<int8_t> view(data);

        REQUIRE(view.size() == 3);
        REQUIRE(view[0] == -128);
        REQUIRE(view[1] == 0);
        REQUIRE(view[2] == 127);
    }

    SECTION("From uint8_t for hamming") {
        std::vector<uint8_t> data = {0xFF, 0x00, 0xAA};
        VectorView<uint8_t> view(data);

        REQUIRE(view.size() == 3);
        REQUIRE(view[0] == 0xFF);
        REQUIRE(view[1] == 0x00);
    }
}

TEST_CASE("VectorView: Const correctness", "[vector_view]") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    const VectorView<float> view(data);

    SECTION("Const member access") {
        REQUIRE(view.size() == 3);
        REQUIRE(view.empty() == false);
        REQUIRE(view.dimensions() == 3);
    }

    SECTION("Const element access") {
        REQUIRE(view[0] == 1.0f);
        REQUIRE(view.front() == 1.0f);
        REQUIRE(view.back() == 3.0f);
    }

    SECTION("Const iteration") {
        int count = 0;
        for (auto val : view) {
            count++;
            (void)val;
        }
        REQUIRE(count == 3);
    }
}
