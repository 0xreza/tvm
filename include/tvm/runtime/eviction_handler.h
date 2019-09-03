/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief abstract class for handling evictions in the managed cuda api
 * \file eviction_handler.h
 */
#ifndef TVM_RUNTIME_EVICTION_HANDLER_H_
#define TVM_RUNTIME_EVICTION_HANDLER_H_

#include <list>
#include <utility>

namespace tvm {
namespace runtime {

  struct MemBlock {
    bool isfree = false;
    size_t size = 0;
    void* start = nullptr;
    std::string owner = "";
    MemBlock(bool isfree, size_t size, void* start) : isfree(isfree), size(size), start(start) {}
  };

  class EvictionHandler {
  public:
    virtual std::pair<std::list<MemBlock>::const_iterator,
                      std::list<MemBlock>::const_iterator>
                      evict(std::list<MemBlock>& mem, size_t nbytes) = 0;

    std::set<std::string> faux_evictions; // for setting eviction rates
  };

} // namespace runtime
} // namespace tvm

#endif  // TVM_RUNTIME_EVICTION_HANDLER_H_
