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
#include <cstdio>

namespace tvm {
namespace runtime {

  static std::mutex outLock;

  inline long now_in_ms() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  }


  #define LOCKED_LOG(x, y) {                                                   \
    const long time = now_in_ms();                                             \
    if (y == "resnet50") {                                                     \
      outLock.lock();                                                          \
      std::cout << x << " " << time << std::endl;                              \
      outLock.unlock();                                                        \
    }                                                                          \
  }

  #define PRE_LOCKED_LOG(x, y) {                                               \
    if (y == "resnet50") {                                                     \
      outLock.lock();                                                          \
      const long time = now_in_ms();                                           \
      std::cout << x << " " << time << std::endl;                              \
      outLock.unlock();                                                        \
    }                                                                          \
  }

  #define TIMESTAMP(x) {                                                       \
    const long time = now_in_ms();                                             \
    outLock.lock();                                                            \
    std::cout << x << " " << time << std::endl;                                \
    outLock.unlock();                                                          \
  }

  typedef struct Task {
    std::atomic<bool> done; // has the op completed
    std::atomic<bool> nextHasCecked; // has the next op in pipeline checked for completion
    std::function<void(void)> operation;
    Task* previousTask;
    std::string modelname;
    Task(std::function<void(void)> op, Task* prev = nullptr, const std::string& name = "") : operation(std::move(op)), previousTask(prev), modelname(name) {
      done.store(false);
      nextHasCecked.store(false);
    }

    Task(std::function<void(void)> op, const std::string& name) : operation(std::move(op)), previousTask(nullptr), modelname(name) {
      done.store(false);
      nextHasCecked.store(false);
    }

    Task(Task&& other) : done(false), nextHasCecked(false) {
      operation = std::move(other.operation);
      previousTask = other.previousTask;
      modelname = std::move(other.modelname);
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

    bool empty() {
      return tasks_.size() == 0;
    }

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
