# HeavyHittersCache
The HeavyHittersCache is a cache with an eviction policy utilizing a count-min sketch probabilistic data structure (similar to a counting Bloom filter) to estimate long-term usage frequencies.  This eviction policy is useful in cases where long-term usage patterns are stable, but short-term disruptions (or limited cache size/observation window) cause problems for LRU and other eviction policies.  For instance, an N-item LRU cache is of litle use if several keys occur every N+1 requests and the rest of the keys occur at very low frequencies.  In this case, the count-min sketch is very likely to get good frequency estimates for the common keys, allowing for better cache hit rates.  The disadvantage of a HeavyHittersCache is that it reacts more slowly to changes in patterns than an LRU cache would.
 
The HeavyHittersCache gets its name from a one-pass streaming solution to the "Heavy Hitters Problem" of finding the K-most-common items in a stream using O(K) space and a simgle pass over the data stream, where the length of the stream isn't known a-priori.  After having seen N items, the cache is very likely to contain all keys that have a total count of at least N/K.  The cache implements a weighted version of the Heavy Hitters Problem where some items can be treated as more important to cache than others.

The HeavyHittersCache uses a count-min sketch to estimate the total "weight" spent calculating the cached value plus (assuming the same cost to re-calculate) "weight" saved by cache hits.  A min-heap is then used to keep track of the estimated weight cost-and-saved for every cached item, and quickly find the lowest "weight" item in the cache to compare against the estimated "weight" of a key being inserted into the cache.  If the new key has more weight, then the lightest key is evicted from the cache.

The HeavyHittersCache also includes some very useful features:

* Per-item expiry times/durations : useful when recoverable errors shoud be cached for a much shorter period than other items
* Per-item cost estimate: useful if some calculations need to fall back to slower/more costly infrastructure
* Exponential decay half-life for accumulated weight : useful when usage patterns shift over time

For a K-item cache, inserting an item typically takes O(1) time under the assumption the newly inserted item's weight is only slightly more than the weight of the lightest item it is evicting from the cache.  Cache misses take O(1) time.  Cache hits typically take O(1) time due to items typically only moving a small number of steps in the min-heap.  Worst-case cache hits have O(log K) time complexity, as do worst-case lookup failures due to a cached item expiring.

This is implementation is not thread-safe.  Synchronize access.

As far as I know, this is the only cache to employ a probabilistic data structure known as a count-min sketch in its eviction strategy.  However, the idea is pretty obvious once you've seen streaming solutions to the "Heavy Hitters Problem", so I wouldn't be surprised if it is used elsewhere.
