/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef TVM_RUNTIME_MULTITENANT_MANAGER_H_
#define TVM_RUNTIME_MULTITENANT_MANAGER_H_

#include <tvm/runtime/executor.h>
#include <map>

namespace tvm {
namespace runtime {

  class Manager {
  public:
    Manager() : disk_(false, true),
                cpu_(false, false),
                load_to_device_(true, false),
                upload_inputs_(true, false),
                gpu_(true, false),
                d2h_pcie_(true, true) {}

    void infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output,
                const std::function<void(void)>& callback) {
      modelLock_.lock();
      if (models_.count(name)) {
        auto& model = models_[name];
        modelLock_.unlock();
        infer_(model, inputName, input, outIndex, output, callback);
      } else {
        modelLock_.unlock();
        loadFromDiskAndInfer_(name, inputName, input, outIndex, output, callback);
      }
    }
  private:
    void loadFromDiskAndInfer_(const std::string& name, const std::string& inputName,
                      DLTensor* input, int outIndex, DLTensor* output,
                      const std::function<void(void)>& callback);

    void infer_(Module& model, const std::string& inputName,
                  DLTensor* input, int outIndex, DLTensor* output,
                  const std::function<void(void)>& callback);

    // map from name to model runtime
    std::map<std::string, Module> models_;
    std::mutex modelLock_;

    // Executors for each resource and a lock to ensure correct ordering
    std::mutex execLock_;
    Executor disk_; // loading model files from disk
    Executor cpu_; // preprocessing model

    // two executors for h2d_pcie
    Executor load_to_device_; // upload model params, setup functions, etc..
    Executor upload_inputs_; // copy input for inference

    Executor gpu_; // execute on gpu
    Executor d2h_pcie_; // copy outputs to host
  };

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MULTITENANT_MANAGER_H_
