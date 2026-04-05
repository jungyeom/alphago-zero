#pragma once

#include <vector>
#include <cstddef>

namespace alphago {

// Forward declaration
struct MCTSNode;

/**
 * Arena allocator for MCTSNode objects.
 *
 * Problem: Each MCTS search creates ~4000+ tree nodes via individual
 * malloc calls. In multi-threaded search_parallel, thread-local malloc
 * arenas accumulate freed-but-unreturned memory, causing RSS to grow
 * ~800MB per iteration.
 *
 * Solution: Allocate nodes from contiguous chunks. When the search
 * completes, reset() "frees" everything at once (O(1), no per-node free,
 * no heap fragmentation).
 */
class NodePool {
public:
    explicit NodePool(size_t chunk_size = 8192);
    ~NodePool();

    /// Allocate one default-initialized MCTSNode.
    MCTSNode* alloc();

    /// Reset — invalidates all previously returned pointers.
    void reset();

    size_t size() const { return total_allocated_; }

private:
    struct Chunk {
        MCTSNode* data;
        size_t capacity;
        size_t used;
    };

    void add_chunk(size_t capacity);

    std::vector<Chunk> chunks_;
    size_t default_chunk_size_;
    size_t total_allocated_;
};

} // namespace alphago
