/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef TVM_RUNTIME_MULTITENANT_MANAGER_H_
#define TVM_RUNTIME_MULTITENANT_MANAGER_H_

#include <tvm/runtime/executor.h>
#include <tvm/runtime/managed_cuda_device_api.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>

namespace tvm {
namespace runtime {

  static int kEvictionRate = 0;

  extern std::mutex outLock;

  class Manager : public EvictionHandler {
  public:
    Manager() : disk_(false),
                cpu_(true),
                load_to_device_(false),
                upload_inputs_(false),
                gpu_(false, 8),
                d2h_pcie_(false),
                out_proc_(true) {
      ManagedCUDADeviceAPI::Global()->SetEvictionHandler(this);
    }

    typedef std::pair<std::list<MemBlock>::const_iterator, std::list<MemBlock>::const_iterator> it_pair;
    it_pair evict(std::list<MemBlock>& mem, size_t nbytes) final {
      // find the shortest range with the memory we need and longest average
      // time since it was used
      auto evict_time = std::chrono::high_resolution_clock::now();
      it_pair ret = std::make_pair(mem.end(), mem.end());
      int avg_time = INT_MIN;

      // for each block, find the smallest subset of blocks covering it and
      // giving enough memory
      auto first = mem.begin();
      for (; first != mem.end(); first++) {
        size_t total = first->size;
        size_t count = 0;
        int total_time = 0;

        if (!first->isfree) {
          count++;
          mapLock_.lock();
          total_time = std::chrono::duration_cast<std::chrono::milliseconds>(evict_time - models_[first->owner].last_use).count();
          mapLock_.unlock();
        }

        auto second = std::next(first);
        for (; second != mem.end(); second++) {
          if (total < nbytes) {
            total += second->size;
            if (!second->isfree) {
              count++;
              mapLock_.lock();
              total_time = std::chrono::duration_cast<std::chrono::milliseconds>(evict_time - models_[second->owner].last_use).count();
              mapLock_.unlock();
            }
          } else {
            break;
          }
        }

        if (total >= nbytes) {
          int new_avg = total_time / count;
          if (new_avg > avg_time) {
            avg_time = new_avg;
            ret = std::make_pair(first, second);
          }
        } else {
          break; // if we couldn't find a set of blocks large enough, then we
                 // won't ever and we can let the caller handle this
        }
      }

      // wait for models to finish running if in use then mark them as evicted
      int count = 0;
      for (auto it = ret.first; it != ret.second; it++) {
        count++;
        if (!it->isfree) {
          mapLock_.lock();
          std::mutex& mlock = modelLocks_[it->owner];
          Model& model = models_[it->owner];
          mapLock_.unlock();

          mlock.lock();
          model.status = ModelStatus::EVICTED;
          model.GetFunction("evicted")();
          mlock.unlock();
        }
      }

      return ret;
    }

    std::future<void> loadModel(const std::string& name, const std::string& source);

    std::future<void> infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {

      //PRE_LOCKED_LOG("TASK queue_inf", name)

      while (!load_to_device_.empty()) {
        usleep(1000);
      } // wait until all evictions have passed

      mapLock_.lock(); // lock the maps because we have to read/write

      while (!modelLocks_[name].try_lock()) {
        mapLock_.unlock();
        mapLock_.lock();
      }

      // here we hold the model lock and the map lock

      CHECK(models_.count(name) == 1) << "Model " << name << " does not exist";
      auto& model = models_[name];
      mapLock_.unlock(); // not using either map now

      model.last_use = std::chrono::high_resolution_clock::now();

      //LOCKED_LOG("TASK_END", name);

      if (rand() % 100 < kEvictionRate) {
        model.status = ModelStatus::EVICTED;
        model.GetFunction("evicted")();
        faux_evictions.insert(name);
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::READY) {
        return infer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::EVICTED) {
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else {
        CHECK(false) << "We've gotten the lock for a model while it was in use.";
      }
    }

  private:

    enum ModelStatus {
      READY = 0,
      IN_USE,
      EVICTED
    };

    struct Model {
      Module mod;
      std::string name;
      ModelStatus status;
      std::chrono::time_point<std::chrono::high_resolution_clock> last_use;

      PackedFunc GetFunction(const std::string& name) {
        return mod.GetFunction(name);
      }

      // need this for compilation, but it should never actually be called
      Model () { CHECK(false) << "Model instantiated without actual code"; }

      Model(Module&& module, const std::string& n) : mod(module),
                                                     name(n),
                                                     status(ModelStatus::READY),
                                                     last_use(std::chrono::high_resolution_clock::now()) {}
    };

    std::future<void> loadToGPUAndInfer_(Model& model, const std::string& inputName,
                  DLTensor* input, int outIndex, DLTensor* output);

    std::future<void> infer_(Model& model, const std::string& inputName, DLTensor* input,
                  int outIndex, DLTensor* output);

    // map from name to model and lock
    std::map<std::string, Model> models_;
    std::map<std::string, std::mutex> modelLocks_;

    // lock for both models_ and modelLocks_ since we have reads and writes to
    // both on multiple threads
    std::mutex mapLock_;

    // model loaded to cpu in separate pipeline
    std::mutex cpuExecLock_;
    Executor disk_; // loading model files from disk
    Executor cpu_; // preprocessing model

    // Executors for each resource and a lock to ensure correct ordering
    std::mutex execLock_;
    // two executors for h2d_pcie
    Executor load_to_device_; // upload model params, setup functions, etc..
    Executor upload_inputs_; // copy input for inference

    Executor gpu_; // execute on gpu
    Executor d2h_pcie_; // copy outputs to host
    Executor out_proc_; // any post processing needed after output is copied back

    // temp storage for loading from disk
    std::mutex sourceLock_;
    std::map<std::string, std::tuple<std::string*, std::string*, Module>> modelSource_;
  };

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MULTITENANT_MANAGER_H_
