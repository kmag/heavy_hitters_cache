#ifndef HEAVY_HITTERS_CACHE_H
#define HEAVY_HITTERS_CACHE_H

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

/* The HeavyHittersCache is a cache employing an eviction strategy useful
 * in cases where long-term usage patterns are
 * stable, but short-term disruptions (or limited cache size/observation
 * window) cause problems for LRU caches.  For instance, an N-item LRU cache is
 * of litle use if several requests occur at frequencies of 1 / 2N and the rest
 * occur at very low frequencies.
 *
 * The HeavyHittersCache gets its name from a one-pass streaming solution to
 * the "Heavy Hitters Problem" of finding the K-most-common items in a stream
 * using O(K) space and a simgle pass over the data stream, where the length of
 * the stream isn't known a-priori.  After having seen N items, the cache is
 * very likely to contain all items that have a total count of at least N/K.
 * The cache implements a weighted version of the problem where some items can
 * be treated as more important to cache than others.
 *
 * As far as I know, this is the only cache to employ a probabilistic data
 * structure known as a count-min sketch (similar to a Bloom filter) in its
 * eviction strategy.  However, the idea is pretty obvious once you've seen
 * streaming solutions to the "Heavy Hitters Problem", so I wouldn't be
 * surprised if it is used elsewhere.
 *
 * The HeavyHittersCache usas probabilistic data struckture known as a
 * count-min sketch to estimate the total "weight" spent calculating the cached
 * value plus (assuming the same cost to re-calculate) "weight" saved by cache
 * hits.  A min-heap is then used to keep track of the estimated weight
 * cost-and-saved for every cached item, which is used for cache eviction purposes.
 *
 * The HeavyHittersCache also includes some very useful featurs:
 *   * Per-item expiry times/durations : useful when recoverable errors
 *       shoud be cached for a much shorter period than other items.
 *   * Per-item cost estimate: useful if some calculations need to
 *       fall back to slower/more costly infrastructure.
 *   * Exponential decay half-life for accumulated weight : useful
 *       when usage patterns shift over time.
 *
 * For a K-item cache, inserting an item typicall takes O(1) time under the
 * assumption the newly inserted item's weight is only slightly more than
 * the weight of the lightest item it is evicting from the cache.
 * Cache misses take O(1) time.  Cache hits typically take O(1) time due to
 * items typically only moving a small number of steps in the min-heap.
 * Worst-case cache hits have O(log K) time complexity, as do worst-case
 * lookup failures due to a cached item expiring.
 *
 * This is implementation is not thread-safe.  Synchronize access.
 */

#include <chrono>
#include <unordered_map>
#include <vector>

#include "sketches/count_min.h"

template<typename Key, typename Value, typename Weight, typename Hash = std::hash<Key>, typename Pred = std::equal_to<Key> >
class HeavyHittersCache {
  public:
    typedef std::chrono::steady_clock::time_point Timestamp;
    typedef std::chrono::steady_clock::duration   Duration;

    HeavyHittersCache(size_t max_size, Duration half_life, Timestamp now = std::chrono::steady_clock::now())
        : next_weight_reduction_(now + half_life / 4), max_size_(max_size), last_weight_reduction_(now),
        half_life_(half_life) {
        heap_.reserve(max_size);
    }
    HeavyHittersCache(const HeavyHittersCache& cache) = delete;

    class CacheEntry {
      public:
        CacheEntry(const Value& value, Timestamp expiry, size_t heap_index) :
            value_(value), expiry_(expiry), heap_index_(heap_index) {
        }
        const Value& value()  const { return value_; }
        Timestamp    expiry() const { return expiry_; }
      private:
        friend class HeavyHittersCache<Key, Value, Weight>;
        Value  value_;
        Timestamp expiry_;
        size_t  heap_index_;
    };

    // We use HeapEntry instead of pair<Key, Weight> to make values exposed to other classes
    // not mutable by other classes, so they can be more safely shared.
    class HeapEntry {
      public:
        HeapEntry(std::pair<const Key, CacheEntry>* entry_ptr, Weight weight, Weight calculation_cost) :
            entry_ptr_(entry_ptr), weight_(weight), calculation_cost_(calculation_cost) {
        }
        const Key&   key()    const { return entry_ptr_->first; }
        const Weight weight() const { return weight_; }
      private:
        friend class HeavyHittersCache<Key, Value, Weight>;
        void SyncIndex(size_t my_index) {
            entry_ptr_->second.heap_index_ = my_index;
        }
        // entry_ptr_ is the address of of cache_.find(key), used to keep cache_ and heap_ consistent.
        std::pair<const Key, CacheEntry>* entry_ptr_;
        Weight weight_;
        Weight calculation_cost_;
    };

    typedef typename std::unordered_map<Key, CacheEntry, Hash, Pred>::iterator iterator;

    iterator end() { return cache_.end(); }

    // This member function is the one to use for looking up cached values.  Its iterator invalidation
    // rules are the same as those for the underlying std::unordered_map.  Note however, that if
    // a cached item expires, subsequent find() operations will cause cache eviction and iterator invalidation.
    iterator find(const Key& key) { return find(key, std::chrono::steady_clock::now()); }

    // The returned iterator is invalidated when a cache Insert evicts the recturned item,
    // the returned item is erase()d, or a later find() operation triggers erasure
    // due to expiry.
    iterator find(const Key& key, Timestamp now) {
        ExponentialDecay(now);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return end();
        }

        if (it->second.expiry() <= now) {
            // Expired, so remove elmeent.
            erase(it);
            return end();
        }

        HeapEntry& original = heap_[it->second.heap_index_];
        original.weight_ += original.calculation_cost_;
        if (original.weight() < original.calculation_cost_) { // Numeric overflow check
            ReduceTotals();
            original.weight_ = Sketches::reduce_counter(std::numeric_limits<Weight>::max());
        }

        // Maintain heap invariant after increasing weight.
        // Note that this won't invalidate any iterators for cache_.
        HeapDown(it->second.heap_index_);

        return it;
    }

    void erase(iterator it) {
        if (it == end()) { return; }

        // Overwirte element with last element in heap and then restore heap invariants.
        size_t heap_index = it->second.heap_index_;
        count_min_.EnsureTotalAtLeast(it->first, heap_[heap_index].weight());  // Flush weight updates back to estimator.
        size_t last_heap_index = heap_.size() - 1;
        if (heap_index == last_heap_index) {
            heap_.pop_back();
            cache_.erase(it);
        } else {
            MoveHeapEntry(heap_index, last_heap_index);
            heap_.pop_back();
            cache_.erase(it);
            HeapDown(heap_index);
        }
    }

    void erase(const Key& key) {
        erase(find(key));
    }

    // Insert new element into cache with absolute expiry time.
    void Insert(const Key& key, const Value& value, Weight calculation_cost, Timestamp expiry,
            Timestamp now = std::chrono::steady_clock::now()) {
        ExponentialDecay(now);
        auto found = cache_.find(key);
        if (found == cache_.end()) {
            // key is not in our exact mapping, so use count_min_ to get a good estimate.
            Weight weight = count_min_.Increment(key, calculation_cost);
            if (weight == std::numeric_limits<Weight>::max()) {
                ReduceTotals();
                // We could look up weight again in count_min_, but it's cheaper to just apply the
                // same operation that count_min_ applied to the value.
                weight = Sketches::reduce_counter(std::numeric_limits<Weight>::max());
            }
            if (heap_.size() < max_size_) {
                size_t next_index = heap_.size();
                cache_.emplace(key, CacheEntry(value, expiry, next_index));
                auto found = cache_.find(key);
                EmplaceBackHeapEntry(&(*found), calculation_cost, weight);
                HeapUp(next_index);
            } else {
                PushPop(key, value, calculation_cost, weight, expiry);  // Possibly evict lightest element
            }
        } else {
            // We have a weight for the element that's exact, except that its initial weight was an estimate.
            size_t found_index = found->second.heap_index_;
            HeapEntry& existing_entry = heap_[found_index];
            existing_entry.calculation_cost_ = calculation_cost;
            Weight weight = existing_entry.weight_ + calculation_cost;
            if (weight < calculation_cost) {
                // Addition has overflowed.
                ReduceTotals();
                weight = Sketches::reduce_counter(std::numeric_limits<Weight>::max());
            }
            existing_entry.weight_ = weight;
            HeapDown(found_index);
        }
    }

    // Insert new element into cache with relative expiry.
    void Insert(const Key& key, const Value& value, Weight calculation_cost, Duration lifetime,
            Timestamp now = std::chrono::steady_clock::now()) {
        Insert(key, value, calculation_cost, now + lifetime, now);
    }

    // With high probabilit, returns a list of the Elements with the highest total Weight seen.
    const std::vector<HeapEntry>& GetContents() const { return heap_; }

    // Returns the estimated total accumulated Weight from cache hits and cache inserts for a given key.
    Weight GetWeight(const Key& key) const {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return count_min_.GetTotal(key);
        } else {
            return heap_[it->second.heap_index_].weight();
        }
    }

    // Modifies totol Weight of seen Keys, for use in implementing exponential decay.
    // This is public so that exponential decay can be implemented in a manner other than time-weighted.
    // For instance, clients may wish to implement exponential decay based on bytes transferred, or
    // have different decay rates at different times of day.
    void MultiplyWeightsBy(double fraction) {
        count_min_.MultiplyTotalsBy(fraction);
        for (auto it = heap_.begin(); it != heap_.end(); ++ it) {
            it->weight_ = static_cast<Weight>(it->weight_ * fraction);
        }
    }

    // Returns true i.f.f. the all of the invariants for the caches internal state currently hold.
    bool CheckInvariants() const {
        bool result = true;
        if (cache_.size() != heap_.size()) {
            std::cerr << "Cache size " << cache_.size() << " vs " << heap_.size() << std::endl;
            result = false;
        } else {
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                const Key& key = it->first;
                size_t heap_index = it->second.heap_index_;
                auto heap_ptr = heap_[heap_index].entry_ptr_;
                auto cache_ptr = &(*it);
                if (heap_ptr != cache_ptr) {
                    std::cerr << "Back references broken for " << key << "  " << heap_index << std::endl;
                    result = false;
                }
            }

            // Check that heap_ actually implements a min-heap
            for (size_t i = 0; i < heap_.size(); ++i) {
                size_t index = heap_[i].entry_ptr_->second.heap_index_;
                if (index != i) {
                    std::cerr << "Round-tripping heap indexes broken at " << i << std::endl;
                    result = false;
                }
            }
        }
        return result;
    }

  private:
    // Perform exponential decay of weights, if necessary.
    void ExponentialDecay(Timestamp now) {
        if (now >= next_weight_reduction_) {
            double half_lives = static_cast<double>((now - last_weight_reduction_).count()) / half_life_.count();
            MultiplyWeightsBy(std::pow(0.5, half_lives));
            last_weight_reduction_ = now;
            next_weight_reduction_ = now + half_life_ / 4;
        }
    }

    // Used for handling numeric overflow when increasing weights.  Proportionately reduces all weights in the cache
    // and all counters in the count-min-sketch.
    void ReduceTotals() {
        count_min_.ReduceTotals();  // Luckily, this will never break the heap property of heap_.
        for (auto it = heap_.begin(); it != heap_.end(); ++ it) {
            it->weight_ = Sketches::reduce_counter(it->weight());
        }
    }

    // After this operation, heap_[src] is left inconsistent with nothing in cache_ refering to it.
    // Later on, src must be the dst of either another MoveHeapEntry call or an OverwriteHeapEntry call.
    void MoveHeapEntry(size_t dst, size_t src) {
        HeapEntry& entry = heap_[dst];
        entry = heap_[src];
        entry.SyncIndex(dst);
    }

    // Overwrites a heap entry.  If this is used to finish a cycle of MoveHeapEntry calls, then cache_ and heap_
    // will be consistent, but the heap invariant may not hold for heap_.
    void OverwriteHeapEntry(size_t dst, std::pair<const Key, CacheEntry>* entry_ptr, Weight calculation_cost, Weight weight) {
        HeapEntry& entry = heap_[dst];
        entry = HeapEntry(entry_ptr, calculation_cost, weight);
        entry.SyncIndex(dst);
    }

    // Pushes a new heap entry on the heap.  This function does not ensure the heap invariant holds.
    void EmplaceBackHeapEntry(std::pair<const Key, CacheEntry>* entry_ptr, Weight calculation_cost, Weight weight) {
        size_t dst = heap_.size();
        heap_.emplace_back(entry_ptr, calculation_cost, weight);
        heap_[dst].SyncIndex(dst);
    }

    // This is the classic binary heap-up operation, with the addition of modifying _index
    // to correct the heap indexes of all modified keys.
    // This can only be called when heap_ has a non-zero size.
    void HeapUp(size_t start_index) {
        if (start_index == 0) {
            return;  // Unlikely
        }
        size_t parent_index = (start_index - 1) / 2;

        if ( ! (heap_[start_index].weight() < heap_[parent_index].weight()) ) {
            return;  // Fast path; very likely
        }

        // Rather than perform a bunch of pairwise swaps of HeapEntries, we save the first
        // state to be overridden, perfom a bunch of single moves, and then copy the saved_entry
        // to its final place.  This saves up to half of the copying vs. pairwise swaps.
        HeapEntry saved_entry = heap_[start_index];

        // Using target_weight allows us to avoid checking in the inner loop if parent_index is zero.
        Weight target_weight = (std::max)(saved_entry.weight(), heap_[0].weight());
        size_t current_index = start_index;
        while (heap_[parent_index].weight() > target_weight) {
            MoveHeapEntry(current_index, parent_index);
            current_index = parent_index;
            parent_index = (current_index - 1) / 2;
        }

        // We've delayed special-case handling of bubling all the way to the top to here.
        if (saved_entry.weight() < target_weight) {
            MoveHeapEntry(current_index, 0);
            current_index = 0;
        }

        heap_[current_index] = saved_entry;
        heap_[current_index].SyncIndex(current_index);
    }

    // This is the classic binary heap-down operation, with the addition of modifying _index
    // to correct the heap indexes of all modified keys.
    // This can only be called when heap_ has a non-zero size.
    void HeapDown(size_t start_index) {
        size_t parent = start_index;
        size_t left  = 2 * parent + 1;
        size_t right = 2 * parent + 2;

        // Check some early-out conditions.
        if (left >= heap_.size() ) {
            return;  // Early out.  No children.
        }

        // Save state of the starting HeapEntry, so we can perform fewer moves than if we perform
        // pairwise swaps.
        HeapEntry saved_entry = heap_[start_index];

        size_t lowest_child = 0;
        while (true) {
            left  = 2 * parent + 1;
            right = 2 * parent + 2;
            // If this loop shows up in profiles, we can pre-calculate the largest parent
            // for which all children are present, and put a fast-path loop before this loop
            // with a loop without bounds checks for left and right child indexes.
            lowest_child = parent;
            if (right < heap_.size() && heap_[right].weight() < saved_entry.weight()) {
                // Prefer to use right child if they have equal weight
                if (heap_[left].weight() < heap_[right].weight()) {
                    lowest_child = left;
                } else {
                    lowest_child = right;
                }
            } else if (left < heap_.size() && heap_[left].weight() < saved_entry.weight()) {
                lowest_child = left;
            } else {
                break;  // We've found where key and weight will be stored.
            }
            HeapEntry& heap_entry = heap_[parent];
            heap_entry = heap_[lowest_child];
            heap_entry.SyncIndex(parent);
            parent = lowest_child;
        }

        heap_[parent] = saved_entry;
        heap_[parent].SyncIndex(parent);
    }

    // This function pops the root of heap_ and replaces it with a new HeapEntry.
    // This can only be called when heap_ has a non-zero size.
    void PushPop(const Key& key, const Value& value, Weight weight, Weight calculation_cost, Timestamp expiry) {
        // This early exit condition looks unintuitive because it properly handles NaN and other cases where Weight
        // might only implement a partial ordering.
        if ( ! (heap_[0].weight() < weight) ) {
            return;
        }
        // We haven't been updating increments in count_min_ for items that are
        // in the K-heaviest heap, so we need to flush the count back to count_min_
        // for the root, because it's about to be evicted.
        count_min_.EnsureTotalAtLeast(heap_[0].key(), heap_[0].weight());
        cache_.erase(heap_[0].key());

        cache_.emplace(key, CacheEntry(value, expiry, 0));
        OverwriteHeapEntry(0, &(*cache_.find(key)), calculation_cost, weight);

        HeapDown(0);
    }

    // Fields ordered in reducing access frequency in order to try and maximize cache locality.
    Timestamp next_weight_reduction_;
    std::unordered_map<Key, CacheEntry, Hash, Pred> cache_;

    // heap_ implements a binary min-heap with the minimum item at heap_[0].
    // We don't use the heap functions in <alrogithms> because as we move nodes to maintain the min-heap invariant,
    // we also need to update accounting information in cache_.  We hold values instead of unique_ptrs here for
    // better cache locality, fewer pointer chases, and because the assumption is that usage patterns change
    // slowly over time, resulting in many more updates of node weight than rea-arangement of node order.
    std::vector<HeapEntry> heap_;
    size_t max_size_;
    CountMinSketch<Key, Weight, Sketches::MultiHash<Key, Hash> > count_min_;
    Timestamp last_weight_reduction_;
    Duration  half_life_;
};

#endif // HEAVY_HITTERS_CACHE_H
