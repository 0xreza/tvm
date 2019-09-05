#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <exception>
#include <chrono>
#include <future>
#include <mutex>
#include <queue>


#ifndef TVM_RUNTIME_BASIC_THREADPOOL_H
#define TVM_RUNTIME_BASIC_THREADPOOL_H

namespace tvm {
namespace runtime {

  class ThreadPool {
  public:
    ThreadPool(int nWorkers) {
      for (int i = 0; i < nWorkers; i++) {
        workers.push_back(std::thread(std::bind(&ThreadPool::worker, this, i)));
      }
    }

    ~ThreadPool() {
      queueLock.lock();
      while (!work.empty()) work.pop();
      exit = true;
      queueLock.unlock();
      cv.notify_one();
      for (auto& worker : workers) {
        worker.join();
      }
    }

    template<typename Ret>
    std::future<Ret> push(std::function<Ret(void)>& f) {
      auto func_promise = std::make_shared<std::promise<Ret>>();
      auto func = [=]() {
        f();
        func_promise->set_value();
      };

      queueLock.lock();
      work.push(std::move(func));
      queueLock.unlock();

      cv.notify_one();
      return func_promise->get_future();
    }

    template<typename Ret, typename... Args>
    std::future<Ret> push(std::function<Ret(Args...)>& f, Args... rest) {
      auto func_promise = std::make_shared<std::promise<Ret>>();
      auto func = [=]() {
        func_promise->set_value(f(rest...));
      };

      queueLock.lock();
      work.push(std::move(func));
      queueLock.unlock();

      cv.notify_one();
      return func_promise->get_future();
    }

    void worker(int id) {
      // separate streams for everybody rn
      // constants for right now
      const int device_type = kDLGPU;
      const int device_id = 0;

      // set stream for the executor
      TVMStreamHandle strm;
      TVMStreamCreate(device_type, device_id, &strm);
      TVMSetStream(device_type, device_id, strm);

      while (true) {
        std::unique_lock<std::mutex> lk(queueLock);

        while (work.size() == 0) {
          if (!exit) {
            cv.wait_for(lk, std::chrono::seconds(1));
          } else if (exit) {
            lk.unlock();
            cv.notify_all();
            return;
          }
        }

        // we have the lock and there is work in the queue
        auto function = std::move(work.front());
        work.pop();
        lk.unlock();

        function();
      }
    }
  private:
    std::vector<std::thread> workers;
    std::queue<std::function<void(void)>> work;
    bool exit = false;

    std::mutex queueLock;
    std::condition_variable cv;
  };

} // namespace runtime
} // namespace tvm

#endif  // TVM_RUNTIME_BASIC_THREADPOOL_H
