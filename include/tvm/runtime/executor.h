/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef TVM_RUNTIME_MULTITENANT_EXECUTOR_H_
#define TVM_RUNTIME_MULTITENANT_EXECUTOR_H_

#include <tvm/runtime/packed_func.h>
#include <queue>
#include <thread>
#include <mutex>

namespace tvm {
namespace runtime {

  static std::mutex outLock;

  inline long now_in_ms() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  }


  #define LOCKED_LOG(x) {                                                      \
    outLock.lock();                                                            \
    std::cout << x << " " << now_in_ms() << std::endl;                         \
    outLock.unlock();                                                          \
  }

  typedef struct Task {
    std::atomic<bool> done; // has the op completed
    std::atomic<bool> nextHasCecked; // has the next op in pipeline checked for completion
    std::function<void(void)> operation;
    Task* previousTask;
    Task(std::function<void(void)> op, Task* prev = nullptr) : operation(std::move(op)), previousTask(prev) {
      done.store(false);
      nextHasCecked.store(false);
    }

    Task(Task&& other) : done(false), nextHasCecked(false) {
      operation = std::move(other.operation);
      previousTask = other.previousTask;
    }
  } Task;

  /*!
   * \brief abstraction for threading based on resource usage
   *
   * Each executor has its own thread (and own local stream) so that it can
   * execute an individual operation and synchronize on it before
   */
  class Executor {
  public:
    Executor(bool sync, bool isLast = false);

    virtual ~Executor() {
      keepGoing_.store(false);
    }

    Task* addTask(Task& t);

  private:
    std::queue<Task> tasks_;
    std::mutex taskMutex_;
    std::atomic<bool> keepGoing_;
    bool syncAfterOp_;
    bool isLast_;
    std::thread runner_;

    void run();
  };

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MULTITENANT_EXECUTOR_H_
