#ifndef COUNT_MIN_SKETCH_H
#define COUNT_MIN_SKETCH_H

/* Copyright 2021 Karl A. Magdsick <karl@magdsick.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* A count-min sketch is a probabilistic data structure similar to a
 * Bloom filter.  It allows us to estimate counts (or other monotonically
 * increasing totals) for a large number of items without using a lot
 * of storage space.
 */
#include <cmath>
#include <limits>
#include <vector>

namespace Sketches {
    // MultiHash is a function object attempting to calculate K uncorrelated
    // hash functions from a given hash function (defaulting to std::hash<T>).
    template<typename T, typename Hash = std::hash<T> >
    class MultiHash {
      public:
        std::vector<uint64_t> operator()(const T& t, size_t hash_count) {
            // Ideally, we'd use a family of hash functions, such as treating the input data
            // as a polynomial in some finite field (say a 64-bit Galois field or a field
            // modulo a 64-bit prime), and then either evaluete the polynomial at
            // hash_count different points, or else use hash_count different fields (differerent
            // feedback polynomials or different modulii).  In the case of a 32-bit Galois field,
            // this is a generalization of the CRC-32 checksum (except using a primitive feedback
            // polynomial).
            //
            // However, it's more widely applicable to just use a good mixing function and hope
            // that std::hash really gets 32 or 64 bits of entropy.
            std::vector<uint64_t> result;
            result.reserve(hash_count);

            uint64_t counter = Hash{}(t);
            // This is the SplitMix64 algorithm used for expanding 64-bit seeds into
            // initial state for xorishiro pseudorandom number generators.
            while (result.size() < hash_count) {
                uint64_t hash = (counter += 0x9e3779b97f4a7c15uLL);
                hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9uLL;
                hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebuLL;
                result.push_back(hash ^ (hash >> 31));
            }

            return result;
        }
    };

    // The implementatino details here are subject to change, but this will always be how
    // a single count is updated by ReduceTotals
    template<typename T>
    T reduce_counter(T t) { return t - (t >> 3); } // multiply by 7/8
}

template<typename T, typename U, typename MultiHasher = Sketches::MultiHash<T> >
class CountMinSketch {
  public:
    // If N is the sum of all "amount" values passed to Increment(), then
    // delta is the probability that Get(T) will exceed the true sum for T by more than epsilot * N;
    CountMinSketch(double epsilon = 1e-4, double delta = 1e-6) {
        hash_count_ = static_cast<size_t>(std::ceil(std::log(1.0 / delta)));
        size_t row_width = static_cast<size_t>(std::ceil(M_E / epsilon));
        mask_ = row_width - 1;
        // Now round mask_ up to one less than the next largest power of two
        mask_ |= (mask_ >> 1);
        mask_ |= (mask_ >> 2);
        mask_ |= (mask_ >> 4);
        mask_ |= (mask_ >> 8);
        mask_ |= (mask_ >> 16);
        mask_ |= (mask_ >> 32);

        counters_.resize((mask_+1) * hash_count_);
    }

    CountMinSketch(const CountMinSketch& sketch) = delete;

    U GetTotal(const T& key) const {  // Gets an estimate of the total amonut T has been incremented by.
        return GetMinCounter(MultiHasher()(key, hash_count_));
    }

    U Increment(const T& key, U amount) {
        // Each counter is an upper bound on the true sum for key.  Take the min.
        std::vector<uint64_t> hashes = MultiHasher()(key, hash_count_);

        U result = amount + GetMinCounter(hashes);
        if (result < amount) {
            result = std::numeric_limits<U>::max();  // Addition overflowed
        }

        EnsureEachCounterAtLeast(hashes, result);
        return result;
    }

    void EnsureTotalAtLeast(const T& key, U value) {
        std::vector<uint64_t> hashes = MultiHasher()(key, hash_count_);
        EnsureEachCounterAtLeast(hashes, value);
    }

    // One way clients might deal with saturating counters is to proportionately decrease all counters when
    // saturation occurs.  In that case, the client can call ReduceTotals().
    void ReduceTotals() {
        for (auto it = counters_.begin(); it != counters_.end(); ++ it) {
            *it = Sketches::reduce_counter(*it);
        }
    }

    // Used for implementing exponential decay of estimates.
    void MultiplyTotalsBy(double fraction) {
        for (auto it = counters_.begin(); it != counters_.end(); ++ it) {
            *it = static_cast<U>(*it * fraction);
        }
    }

  private:
    // For each of the given hash values, ensures that the counter for the corresponding hash function is
    // at least the given value.
    inline void EnsureEachCounterAtLeast(const std::vector<uint64_t>& hashes, U value) {
        size_t base = 0;
        size_t stride = mask_ + 1;  // base is incremented by stride to look at counters for the next hash function
        for (auto it = hashes.cbegin(); it != hashes.cend(); ++it, base += stride) {
            size_t index = base + (mask_ & *it);
            if (counters_[index] < value) {
                counters_[index] = value;
            }
        }
    }

    // Gets an estimate of the total amonut T has been incremented by.
    inline U GetMinCounter(const std::vector<uint64_t>& hashes) const {
        // Each counter is an upper bound on the true sum for key.  Take the min.
        U result = std::numeric_limits<U>::max();
        const size_t stride = mask_ + 1;  // Amount to increment "base" to get to the next hash function's counters
        size_t base = 0;
        for (auto it = hashes.cbegin(); it != hashes.cend(); ++it, base += stride) {
            size_t index = base + (mask_ & *it);
            result = (std::min)(result, counters_[index]);
        }
        return result;
    }
    size_t mask_;  // The number of counters per row is always a power of two, and mask_ is one less.
    size_t hash_count_;  // The number of independent hash functions
    std::vector<U> counters_;
};

#endif // COUNT_MIN_SKETCH_H
