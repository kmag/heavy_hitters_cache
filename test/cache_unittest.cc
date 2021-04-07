#include <ctime>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "heavy_hitters_cache.h"

using namespace std;

static void TestSimpleCase();
static void TestHeavyHittersInvariants(size_t cache_size, size_t key_count, uint64_t seed);

int main() {
    TestSimpleCase();

    TestHeavyHittersInvariants(10,        20, 0);
    TestHeavyHittersInvariants(10,     10000, 1);
    TestHeavyHittersInvariants(100,   100000, 2);

    std::random_device true_random;
    TestHeavyHittersInvariants(1000,  100000, true_random() + (static_cast<uint64_t>(true_random()) << 32));

    return 0;
}

// Helper to check that the given heap really is organized as a binary heap.
template<typename T>
static void CheckHeapInvariant(const vector<T>& heap) {
    for (size_t child = heap.size() - 1; child > 0; -- child) {
        size_t parent = (child - 1) / 2;
        EXPECT_LE( heap[parent].weight(), heap[child].weight() );
    }
}

template<typename T, typename U, typename V, typename W>
static void CheckHeavyHittersInvariants(const HeavyHittersCache<T, U, V>& cache,
        const unordered_map<T, W>& actual_counts) {

    // cache.find() will re-order the internal heap, breaking tests that call find while
    // iterating over it, so we make a copy rather than holding on to the reference.
    auto heaviest = cache.GetContents();
    std::unordered_map<T, V> heaviest_counts;
    for (auto it = heaviest.cbegin(); it != heaviest.cend(); ++it) {
        heaviest_counts.emplace(it->key(), it->weight());
    }

    // The fact that heaviest is a heap is an implementation detail that might change in the future,
    // but it's an important invariant for the correctness of the current implementation.
    CheckHeapInvariant(heaviest);

    uint64_t total_weight = 0;
    for (auto it = actual_counts.cbegin(); it != actual_counts.cend(); ++ it) {
        total_weight += it->second;
    }
    uint64_t cutoff_weight = (total_weight + heaviest.size() - 1) / heaviest.size();

    for (auto it = heaviest_counts.cbegin(); it != heaviest_counts.cend(); ++ it) {
        // Check that all of the heaviest hitter keys were actually inserted.
        EXPECT_NE(actual_counts.find(it->first), actual_counts.end());
    }

    // In count-min-sketch terms, error_bound is epsion times the total weight.  There's a 1.0e-6 probability
    // a given key will be over-estimated by more than that amount.
    const uint32_t error_bound = static_cast<uint32_t>(std::ceil(1.0e-4 * total_weight));

    for (auto actual = actual_counts.cbegin(); actual != actual_counts.cend(); ++actual) {
        V estimated_weight = cache.GetWeight(actual->first);
        // No weights should be under-estimated
        EXPECT_GE(estimated_weight, actual->second);
        // Over-estimates should be within the error bounds
        EXPECT_LE(estimated_weight, actual->second + error_bound);

        auto estimated = heaviest_counts.find(actual->first);
        if (estimated == heaviest_counts.end()) {
            // For all inserted keys, if it isn't in the cache, it should be lighter than the total weight
            // divided by the size of the cache.
            EXPECT_LE(actual->second, cutoff_weight + error_bound);
        } else {
            // No weights should be under-estimated
            EXPECT_GE(estimated->second, actual->second);
            // Over-estimates should be within the error bounds
            EXPECT_LE(estimated->second, actual->second + error_bound);
        }
    }
}

static void TestHeavyHittersInvariants(size_t cache_size, size_t key_count, uint64_t seed) {
    cerr << "Testing HeavyHittersCache of size " << cache_size << " with " << key_count <<
        " keys and seed of " << seed << endl;

    auto now = chrono::steady_clock::now();
    auto lifetime = chrono::hours(1);
    HeavyHittersCache<string, uint64_t, uint64_t> cache(cache_size, chrono::hours(12), now);
    unordered_map<string, uint32_t> expected_counts;
    mt19937_64 prng(seed);
    vector<pair<string, uint64_t> > data;  // There's a smal chance of containing dupblicates, but that won't break the test.

    data.reserve(key_count);
    while (data.size() < key_count) {
        uint64_t value = prng();
        string key = to_string(value);
        data.emplace_back(key, value);
    }

    for (uint64_t i = (std::min)((std::max)( key_count * 1000uLL, 100000uLL), 1000000uLL); i > 0; --i) {
        size_t rand_index = prng() % data.size();
        const string& key = data[rand_index].first;
        ++ expected_counts[key];  // Counter default-constructed to zero if it doesn't exist yet.
        if (cache.find(key, now) == cache.end()) {
            cache.Insert(key, data[rand_index].second, 1, lifetime, now);  // Insert after cache miss.
        }
    }
  
    CheckHeavyHittersInvariants(cache, expected_counts);

    EXPECT_TRUE( cache.CheckInvariants() );

    // We reerse-iterate here because we know internal implementation details of the min-heap we're
    // iterating over. cache.find() is going to increase weights, but reverse iteration means that
    // children always increased before their parents, so elements are never re-arranged in the heap.
    for (auto it = cache.GetContents().crbegin(); it != cache.GetContents().crend(); ++it) {
        auto found = cache.find(it->key(), now);
        EXPECT_NE( found, cache.end() );
        if (found != cache.end()) {
            EXPECT_EQ(it->key(), found->first);
            EXPECT_EQ(found->first, to_string(found->second.value()));
        }
    }
}

// A simple hand-written test case.
static void TestSimpleCase() {
    cerr << "Tisting simple hand-written test case" << endl;

    auto now = chrono::steady_clock::now();
    auto lifetime = chrono::hours(1);
    HeavyHittersCache<string, uint64_t, uint64_t> cache(2, chrono::hours(12), now);

    cache.Insert("zero", 0, 1, lifetime, now);
    cache.find("zero", now);
    cache.find("zero", now);
    EXPECT_EQ( 3, cache.GetWeight("zero"));

    EXPECT_EQ( cache.end(), cache.find("one", now) );
    cache.Insert("one", 1, 1, lifetime, now);
    EXPECT_EQ( 1, cache.GetWeight("one"));
    cache.Insert("one", 1, 1, lifetime, now);

    cache.Insert("two", 1, 1, lifetime, now);
    EXPECT_EQ( 1, cache.GetWeight("two"));
    cache.Insert("two", 1, 1, lifetime, now);
    EXPECT_EQ( 2, cache.GetWeight("two"));
    EXPECT_NE( cache.end(), cache.find("one", now) );
    cache.Insert("two", 1, 1, lifetime, now);
    EXPECT_EQ( 3, cache.GetWeight("two"));
    cache.Insert("two", 1, 1, lifetime, now);
    EXPECT_EQ( 4, cache.GetWeight("two"));

    EXPECT_EQ( 3, cache.GetWeight("zero"));
    EXPECT_EQ( 4, cache.GetWeight("two"));
}
