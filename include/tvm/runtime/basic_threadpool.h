#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <exception>
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
      while (true) {
        std::unique_lock<std::mutex> lk(queueLock);

        if (work.size() == 0) {
          cv.wait(lk);
        }

        if (work.size() == 0) {
          // if the size is still 0 after being woke, time to exit
          cv.notify_one();
          return;
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

    std::mutex queueLock;
    std::condition_variable cv;
  };

} // namespace runtime
} // namespace tvm

#endif  // TVM_RUNTIME_BASIC_THREADPOOL_H
