
#include <dlpack/dlpack.h>
#include <tvm/runtime/manager.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <chrono>

#include "util/shmem.h"

namespace tvm {
namespace runtime {

  const int device_type = kDLGPU;
  const int device_id = 0;

  void Manager::loadFromDiskAndInfer_(const std::string& name, const std::string& inputName,
                    DLTensor* input, int outIndex, DLTensor* output,
                    const std::function<void(void)>& callback) {
    // assign a single task of loading the model from disk, which takes quite a
    // while to load up, then have it trigger enqueueing of all tasks for infer
    Task loadAndEnqueue([=, &callback] {
      LOCKED_LOG("TASK disk_load");
      const PackedFunc load_module(*tvm::runtime::Registry::Get("module.loadfile_so"));

      // Graph structure
      std::ifstream json_in(name + ".json", std::ios::in);  // read as text
      std::string* json_data = new std::string((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
      json_in.close();

      // load params contiguous block
      std::ifstream params_in(name + ".params_contig", std::ios::binary);
      std::string* params_data = new std::string((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
      params_in.close();

      // module containing host and dev code
      Module mod_syslib = load_module(copy_to_memory(name + ".so"), "so");

      // lock on executors and enqueue the tasks
      this->execLock_.lock();

      Task createModel([=]{
        LOCKED_LOG("TASK cpu");
        TVMByteArray params_arr;
        params_arr.data = params_data->c_str();
        params_arr.size = params_data->length();

        const PackedFunc create_runtime(*tvm::runtime::Registry::Get("tvm.decoupled_graph_runtime.create_contiguous"));
        this->modelLock_.lock();
        this->models_.emplace(name, create_runtime(*json_data, mod_syslib, device_type, device_id));
        PackedFunc load_params = this->models_[name].GetFunction("load_params_contig");
        this->modelLock_.unlock();
        load_params(params_arr);
        delete json_data;
        delete params_data;
      });

      Task* prev = this->cpu_.addTask(createModel);
      Task copyToDevice([=] {
        LOCKED_LOG("TASK load_to_device");
        this->modelLock_.lock();
        PackedFunc load = this->models_[name].GetFunction("load_to_device");
        this->modelLock_.unlock();

        load();
      }, prev);

      prev = this->load_to_device_.addTask(copyToDevice);
      Task uploadInput([=] {
        LOCKED_LOG("TASK input");
        this->modelLock_.lock();
        PackedFunc set_input = this->models_[name].GetFunction("set_input");
        this->modelLock_.unlock();

        set_input(inputName, input);
      }, prev);

      prev = this->upload_inputs_.addTask(uploadInput);
      Task run([=] {
        LOCKED_LOG("TASK run");
        this->modelLock_.lock();
        PackedFunc run = this->models_[name].GetFunction("run");
        this->modelLock_.unlock();

        run();
      }, prev);

      prev = this->gpu_.addTask(run);
      Task getOutput([=, &callback] {
        LOCKED_LOG("TASK output");
        this->modelLock_.lock();
        PackedFunc get_output = this->models_[name].GetFunction("get_output");
        this->modelLock_.unlock();

        get_output(outIndex, output);
        std::thread(callback).detach();
      }, prev);

      this->d2h_pcie_.addTask(getOutput);

      this->execLock_.unlock();

    });

    this->execLock_.lock();
    disk_.addTask(loadAndEnqueue);
    this->execLock_.unlock();
  }

  void Manager::infer_(Module& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output,
                const std::function<void(void)>& callback) {
    // lock on executors and enqueue the tasks
    this->execLock_.lock();

    Task uploadInput([=, &model] {
      LOCKED_LOG("TASK input");
      PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
    });

    Task* prev = upload_inputs_.addTask(uploadInput);
    Task run([=, &model] {
      LOCKED_LOG("TASK run");
      PackedFunc run = model.GetFunction("run");
      run();
    }, prev);

    prev = gpu_.addTask(run);
    Task getOutput([=, &model, &callback] {
      LOCKED_LOG("TASK output");
      PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);
      std::thread(callback).detach();
    }, prev);

    this->d2h_pcie_.addTask(getOutput);

    this->execLock_.unlock();
  }

}  // namespace runtime
}  // namespace tvm
