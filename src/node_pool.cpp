#include "node_pool.h"
#include "mcts_node.h"
#include <cstdlib>
#include <new>

namespace alphago {

NodePool::NodePool(size_t chunk_size)
    : default_chunk_size_(chunk_size)
    , total_allocated_(0)
{
    add_chunk(default_chunk_size_);
}

NodePool::~NodePool() {
    for (auto& chunk : chunks_) {
        // Destruct in-place, then free raw memory
        for (size_t i = 0; i < chunk.used; i++) {
            chunk.data[i].~MCTSNode();
        }
        std::free(chunk.data);
    }
}

void NodePool::add_chunk(size_t capacity) {
    // Use mmap-backed allocation for large chunks (glibc uses mmap for >= 128KB
    // by default, and mmap memory is returned to OS on free unlike arena memory)
    void* raw = std::malloc(capacity * sizeof(MCTSNode));
    if (!raw) throw std::bad_alloc();

    Chunk chunk;
    chunk.data = static_cast<MCTSNode*>(raw);
    chunk.capacity = capacity;
    chunk.used = 0;
    chunks_.push_back(chunk);
}

MCTSNode* NodePool::alloc() {
    auto& chunk = chunks_.back();
    if (chunk.used >= chunk.capacity) {
        // Current chunk full — allocate a new one (double size)
        add_chunk(chunks_.back().capacity * 2);
    }

    auto& active = chunks_.back();
    MCTSNode* node = new (&active.data[active.used]) MCTSNode();
    active.used++;
    total_allocated_++;
    return node;
}

void NodePool::reset() {
    // Destruct all nodes but keep the chunk memory allocated for reuse.
    // This avoids repeated malloc/free cycles.
    for (auto& chunk : chunks_) {
        for (size_t i = 0; i < chunk.used; i++) {
            chunk.data[i].~MCTSNode();
        }
        chunk.used = 0;
    }

    // Keep only the first (and possibly grown) chunk to avoid
    // keeping too many chunks around
    if (chunks_.size() > 1) {
        // Free all but the largest chunk
        size_t largest_idx = 0;
        size_t largest_cap = 0;
        for (size_t i = 0; i < chunks_.size(); i++) {
            if (chunks_[i].capacity > largest_cap) {
                largest_cap = chunks_[i].capacity;
                largest_idx = i;
            }
        }
        Chunk keep = chunks_[largest_idx];
        for (size_t i = 0; i < chunks_.size(); i++) {
            if (i != largest_idx) {
                std::free(chunks_[i].data);
            }
        }
        chunks_.clear();
        chunks_.push_back(keep);
    }

    total_allocated_ = 0;
}

} // namespace alphago
