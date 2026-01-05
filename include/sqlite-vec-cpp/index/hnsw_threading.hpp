#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace sqlite_vec_cpp::index {

/// Simple spinlock using atomic flag
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void unlock() { flag_.clear(std::memory_order_release); }
};

/// Thread-local random number generator for parallel operations
/// Avoids contention on shared RNG during parallel layer assignment
class ThreadLocalRNG {
    static thread_local std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_{0.0f, 1.0f};

public:
    explicit ThreadLocalRNG(uint32_t seed = 42) { rng_.seed(seed); }

    [[nodiscard]] float random() {
        float r = dist_(rng_);
        if (r == 0.0f)
            r = 1e-9f;
        return r;
    }

    [[nodiscard]] size_t random_layer(float ml_factor) {
        return static_cast<size_t>(-std::log(random()) * ml_factor);
    }

    [[nodiscard]] uint32_t random_uint(uint32_t max) {
        std::uniform_int_distribution<uint32_t> dist(0, max - 1);
        return dist(rng_);
    }
};

thread_local std::mt19937 ThreadLocalRNG::rng_{std::random_device{}()};

/// Per-node lock array using pointer-based storage to avoid move issues
class NodeLocks {
    std::vector<std::unique_ptr<SpinLock>> locks_;
    size_t capacity_;

public:
    explicit NodeLocks(size_t capacity = 1000) : capacity_(capacity) {
        locks_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            locks_.push_back(std::make_unique<SpinLock>());
        }
    }

    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) {
            return; // Only grow, never shrink
        }

        size_t old_capacity = capacity_;
        capacity_ = new_capacity;

        for (size_t i = old_capacity; i < new_capacity; ++i) {
            locks_.push_back(std::make_unique<SpinLock>());
        }
    }

    [[nodiscard]] size_t capacity() const { return capacity_; }

    SpinLock& get(size_t node_id) { return *locks_[node_id % capacity_]; }

    /// RAII guard for automatic lock/unlock
    class Guard {
        SpinLock* lock_;
        bool owns_lock_;

    public:
        explicit Guard(SpinLock& lock) : lock_(&lock), owns_lock_(true) { lock_->lock(); }

        ~Guard() {
            if (owns_lock_) {
                lock_->unlock();
            }
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        Guard(Guard&& other) noexcept : lock_(other.lock_), owns_lock_(other.owns_lock_) {
            other.owns_lock_ = false;
        }

        Guard& operator=(Guard&& other) noexcept {
            if (this != &other) {
                if (owns_lock_) {
                    lock_->unlock();
                }
                lock_ = other.lock_;
                owns_lock_ = other.owns_lock_;
                other.owns_lock_ = false;
            }
            return *this;
        }
    };

    /// Create guard for specific node
    Guard guard_for(size_t node_id) { return Guard(get(node_id)); }
};

/// Hash-based striped lock manager for HNSW neighbor list modifications
/// Uses consistent hashing to prevent deadlocks (Apache Lucene pattern)
/// Reduces lock contention compared to per-node locking
class StripedLockManager {
    static constexpr size_t DEFAULT_NUM_STRIPES = 512;
    std::vector<std::unique_ptr<SpinLock>> locks_;
    size_t num_stripes_;

public:
    explicit StripedLockManager(size_t num_stripes = DEFAULT_NUM_STRIPES)
        : num_stripes_(num_stripes) {
        locks_.reserve(num_stripes);
        for (size_t i = 0; i < num_stripes; ++i) {
            locks_.push_back(std::make_unique<SpinLock>());
        }
    }

    /// Move constructor
    StripedLockManager(StripedLockManager&& other) noexcept
        : locks_(std::move(other.locks_)), num_stripes_(other.num_stripes_) {
        other.num_stripes_ = 0;
    }

    /// Move assignment
    StripedLockManager& operator=(StripedLockManager&& other) noexcept {
        if (this != &other) {
            locks_ = std::move(other.locks_);
            num_stripes_ = other.num_stripes_;
            other.num_stripes_ = 0;
        }
        return *this;
    }

    /// Delete copy operations
    StripedLockManager(const StripedLockManager&) = delete;
    StripedLockManager& operator=(const StripedLockManager&) = delete;

    void reserve(size_t expected_nodes) { (void)expected_nodes; }

    [[nodiscard]] static size_t hash_node_id(size_t node_id) {
        size_t h = node_id;
        h ^= (h >> 20) ^ (h >> 12);
        return h ^ (h >> 7) ^ (h >> 4);
    }

    [[nodiscard]] static size_t hash(size_t node_id, size_t level) {
        size_t h = level * 31 + node_id;
        h ^= (h >> 20) ^ (h >> 12);
        return h ^ (h >> 7) ^ (h >> 4);
    }

    [[nodiscard]] size_t num_stripes() const { return num_stripes_; }

    SpinLock& get(size_t node_id, size_t level) {
        size_t idx = hash(node_id, level) % num_stripes_;
        return *locks_[idx];
    }

    SpinLock& get_by_id(size_t node_id) {
        size_t idx = hash_node_id(node_id) % num_stripes_;
        return *locks_[idx];
    }

    /// RAII guard for automatic lock/unlock
    class Guard {
        SpinLock* lock_;
        bool owns_lock_;

    public:
        explicit Guard(SpinLock& lock) : lock_(&lock), owns_lock_(true) { lock_->lock(); }

        ~Guard() {
            if (owns_lock_) {
                lock_->unlock();
            }
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        Guard(Guard&& other) noexcept : lock_(other.lock_), owns_lock_(other.owns_lock_) {
            other.owns_lock_ = false;
        }

        Guard& operator=(Guard&& other) noexcept {
            if (this != &other) {
                if (owns_lock_) {
                    lock_->unlock();
                }
                lock_ = other.lock_;
                owns_lock_ = other.owns_lock_;
                other.owns_lock_ = false;
            }
            return *this;
        }
    };

    /// Create guard for node (hash-based stripe selection)
    Guard guard(size_t node_id, size_t level) { return Guard(get(node_id, level)); }

    /// Create guard for node by ID only
    Guard guard_by_id(size_t node_id) { return Guard(get_by_id(node_id)); }
};

/// Thread pool executor for parallel operations
class ThreadPool {
    size_t num_threads_;
    std::vector<std::jthread> threads_;
    std::atomic<size_t> next_idx_{0};

public:
    explicit ThreadPool(size_t threads = 0)
        : num_threads_(threads ? threads : std::thread::hardware_concurrency()) {}

    [[nodiscard]] size_t num_threads() const { return num_threads_; }

    template <typename Func> void parallel_for(size_t count, Func&& func) {
        if (count == 0 || num_threads_ == 1) {
            for (size_t i = 0; i < count; ++i) {
                func(0, i);
            }
            return;
        }

        size_t local_count = count;
        next_idx_.store(0, std::memory_order_relaxed);
        threads_.clear();
        threads_.reserve(num_threads_);

        for (size_t t = 0; t < num_threads_; ++t) {
            threads_.emplace_back([this, &func, t, local_count] {
                while (true) {
                    size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= local_count) {
                        break;
                    }
                    func(t, idx);
                }
            });
        }

        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads_.clear();
    }

    template <typename Func> void parallel_for(size_t count, size_t threads, Func&& func) {
        if (count == 0 || threads == 1) {
            for (size_t i = 0; i < count; ++i) {
                func(0, i);
            }
            return;
        }

        size_t local_count = count;
        next_idx_.store(0, std::memory_order_relaxed);
        threads_.clear();
        threads_.reserve(threads);

        for (size_t t = 0; t < threads; ++t) {
            threads_.emplace_back([this, &func, t, local_count] {
                while (true) {
                    size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= local_count) {
                        break;
                    }
                    func(t, idx);
                }
            });
        }

        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads_.clear();
    }

    ~ThreadPool() {
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.detach();
            }
        }
    }
};

/// Simple thread-safe counter for parallel reductions
template <typename T> class AtomicCounter {
    std::atomic<T> value_{0};

public:
    void add(T val) { value_.fetch_add(val, std::memory_order_relaxed); }

    void increment() { add(1); }

    [[nodiscard]] T load() const { return value_.load(std::memory_order_relaxed); }

    void reset() { value_.store(0, std::memory_order_relaxed); }
};

} // namespace sqlite_vec_cpp::index
