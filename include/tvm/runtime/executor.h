/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef TVM_RUNTIME_MULTITENANT_EXECUTOR_H_
#define TVM_RUNTIME_MULTITENANT_EXECUTOR_H_

#include <tvm/runtime/packed_func.h>
#include <cuda_runtime.h>
#include <tvm/runtime/cuda_common.h>
#include <tvm/runtime/basic_threadpool.h>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdio>

namespace tvm {
namespace runtime {

  extern std::mutex outLock;

  inline long now_in_ms() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  }

  #define LOCKED_LOG_TIME(model, message, time) {                              \
    const long timestamp = now_in_ms();                                        \
    outLock.lock();                                                            \
    std::cout << model << " " << message << " " << time << " " << timestamp << std::endl;\
    outLock.unlock();                                                          \
  }

  #define LOCKED_LOG(x, y) {                                                   \
    const long time = now_in_ms();                                             \
    outLock.lock();                                                            \
    std::cout << y << " " << x << " " << time << std::endl;                    \
    outLock.unlock();                                                          \
  }

  #define PRE_LOCKED_LOG(x, y) {                                               \
    outLock.lock();                                                            \
    const long time = now_in_ms();                                             \
    std::cout << y << " "  << x << " " << time << std::endl;                   \
    outLock.unlock();                                                          \
  }

  #define TIMESTAMP(x) {                                                       \
    const long time = now_in_ms();                                             \
    outLock.lock();                                                            \
    std::cout << x << " " << time << std::endl;                                \
    outLock.unlock();                                                          \
  }

  typedef struct Task {
    std::string task_name_;
    std::atomic<bool> started; // has the op completed
    std::atomic<bool> nextHasCecked; // has the next op in pipeline checked for completion
    std::mutex handleDep;
    std::function<std::function<void(void)>(void)> operation;
    Task* previousTask;
    Task* nextTask = nullptr;
    std::string modelname;
    cudaEvent_t sync; // event for syncing tasks happening on different streams

    void setNextTask(Task* next) {
      nextTask = next;
    }

    Task(const std::string& t_name, std::function<std::function<void(void)>(void)> op, Task* prev = nullptr, const std::string& name = "") : task_name_(t_name), operation(std::move(op)), previousTask(prev), modelname(name) {
      started.store(false);
      nextHasCecked.store(false);
    }

    Task(const std::string& t_name, std::function<std::function<void(void)>(void)> op, const std::string& name) : task_name_(t_name), operation(std::move(op)), previousTask(nullptr), modelname(name) {
      started.store(false);
      nextHasCecked.store(false);
    }

    Task(Task&& other) : started(false), nextHasCecked(false) {
      task_name_  = std::move(other.task_name_);
      operation = std::move(other.operation);
      previousTask = other.previousTask;
      modelname = std::move(other.modelname);
      sync = std::move(other.sync);
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
    Executor(bool isLast = false, int threads = 1);

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
    bool isLast_;
    int nThreads_;
    ThreadPool tp_;
    std::thread runner_;

    static std::atomic<int> global_exec_count;

    void run();
  };

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MULTITENANT_EXECUTOR_H_
