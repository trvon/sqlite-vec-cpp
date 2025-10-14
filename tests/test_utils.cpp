#include <cassert>
#include <iostream>
#include <sqlite-vec-cpp/utils/array.hpp>

using namespace sqlite_vec_cpp;
using namespace sqlite_vec_cpp::utils;

void test_array_basic() {
    std::cout << "Testing Array basic operations..." << std::endl;

    // Create array with initial capacity
    Array<int> arr(10);
    assert(arr.empty());
    assert(arr.capacity() >= 10);

    // Append elements
    auto result = arr.append(42);
    assert(result.has_value());
    assert(arr.size() == 1);
    assert(arr[0] == 42);

    result = arr.append(100);
    assert(result.has_value());
    assert(arr.size() == 2);
    assert(arr[1] == 100);

    // Test growth beyond initial capacity
    for (int i = 0; i < 100; ++i) {
        result = arr.append(i);
        assert(result.has_value());
    }
    assert(arr.size() == 102);

    // Test access
    assert(arr[0] == 42);
    assert(arr[1] == 100);
    assert(arr[2] == 0);

    // Test span
    auto sp = arr.span();
    assert(sp.size() == arr.size());
    assert(sp[0] == 42);

    // Test clear
    arr.clear();
    assert(arr.empty());
    assert(arr.size() == 0);

    std::cout << "  Array basic tests passed!" << std::endl;
}

void test_array_float() {
    std::cout << "Testing Array with float..." << std::endl;

    Array<float> arr;

    std::vector<float> test_data = {1.0f, 2.5f, 3.14f, 4.2f};
    for (float val : test_data) {
        auto result = arr.append(val);
        assert(result.has_value());
    }

    assert(arr.size() == test_data.size());
    for (std::size_t i = 0; i < test_data.size(); ++i) {
        assert(arr[i] == test_data[i]);
    }

    // Test data() pointer
    const float* data_ptr = arr.data();
    assert(data_ptr != nullptr);
    assert(data_ptr[0] == 1.0f);
    assert(data_ptr[3] == 4.2f);

    std::cout << "  Array float tests passed!" << std::endl;
}

void test_array_move() {
    std::cout << "Testing Array move semantics..." << std::endl;

    Array<std::string> arr1;
    arr1.append("hello");
    arr1.append("world");

    // Move construction
    Array<std::string> arr2(std::move(arr1));
    assert(arr2.size() == 2);
    assert(arr2[0] == "hello");
    assert(arr2[1] == "world");

    // Move assignment
    Array<std::string> arr3;
    arr3 = std::move(arr2);
    assert(arr3.size() == 2);
    assert(arr3[0] == "hello");

    std::cout << "  Array move tests passed!" << std::endl;
}

void test_dynamic_array() {
    std::cout << "Testing DynamicArray..." << std::endl;

    DynamicArray arr(sizeof(int), 5);

    int val1 = 42;
    int val2 = 100;

    auto result = arr.append(&val1);
    assert(result.has_value());
    assert(arr.size() == 1);

    result = arr.append(&val2);
    assert(result.has_value());
    assert(arr.size() == 2);

    // Get elements
    const int* retrieved1 = static_cast<int*>(arr.get(0));
    const int* retrieved2 = static_cast<int*>(arr.get(1));

    assert(retrieved1 != nullptr);
    assert(*retrieved1 == 42);
    assert(retrieved2 != nullptr);
    assert(*retrieved2 == 100);

    // Test out of bounds
    assert(arr.get(10) == nullptr);

    arr.clear();
    assert(arr.empty());

    std::cout << "  DynamicArray tests passed!" << std::endl;
}

void test_array_iteration() {
    std::cout << "Testing Array iteration..." << std::endl;

    Array<int> arr;
    for (int i = 0; i < 10; ++i) {
        arr.append(i * 2);
    }

    // Range-based for loop
    int sum = std::accumulate(arr.begin(), arr.end(), 0);
    assert(sum == 90); // 0+2+4+6+8+10+12+14+16+18 = 90

    // Iterator
    int count = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        assert(*it == count * 2);
        ++count;
    }
    assert(count == 10);

    std::cout << "  Array iteration tests passed!" << std::endl;
}

int main() {
    try {
        test_array_basic();
        test_array_float();
        test_array_move();
        test_dynamic_array();
        test_array_iteration();

        std::cout << "\nAll Array/utility tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
