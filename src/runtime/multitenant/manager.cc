
#include <dlpack/dlpack.h>
#include <tvm/runtime/manager.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <chrono>
#include <future>

#include "util/shmem.h"

namespace tvm {
namespace runtime {

  extern std::mutex outLock;

  const int device_type = kDLGPU;
  const int device_id = 0;

  std::future<void> Manager::loadModel(const std::string& name, const std::string& source) {

    auto ret = std::make_shared<std::promise<void>>();

    this->mapLock_.lock();
    this->modelLocks_[name].lock();
    this->mapLock_.unlock();

    cpuExecLock_.lock();

    Task loadFromDisk("disk_load", [=] {
      const PackedFunc load_module(*tvm::runtime::Registry::Get("module.loadfile_so"));

      // load graph structure
      std::ifstream json_in(source + ".json", std::ios::in);  // read as text
      std::string* json_data = new std::string((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
      json_in.close();

      // load params contiguous block
      std::ifstream params_in(source + ".params_contig", std::ios::binary);
      std::string* params_data = new std::string((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
      params_in.close();

      // module containing host and dev code
      std::cout << source + ".so changes to\n";
      Module mod_syslib = load_module(copy_to_memory(source + ".so"), "so");
      std::cout << "and that might not make sense\n";

      sourceLock_.lock();
      modelSource_.emplace(name, std::make_tuple(
          json_data, params_data, std::move(mod_syslib)
      ));
      sourceLock_.unlock();

      return [](){}; // nothing happens on gpu so pass a nop
    }, name);
    Task* prev = disk_.addTask(loadFromDisk);

    Task createModel("cpu", [=]{
      sourceLock_.lock();
      std::string* json_data = std::get<0>(modelSource_[name]);
      std::string* params_data = std::get<1>(modelSource_[name]);
      Module& mod_syslib = std::get<2>(modelSource_[name]);
      sourceLock_.unlock();

      TVMByteArray params_arr;
      params_arr.data = params_data->c_str();
      params_arr.size = params_data->length();

      // create the model and set its status as in use
      const PackedFunc create_runtime(*tvm::runtime::Registry::Get("tvm.decoupled_graph_runtime.create_contiguous"));
      this->mapLock_.lock();
      this->models_.emplace(std::piecewise_construct,
                            std::forward_as_tuple(name),
                            std::forward_as_tuple(create_runtime(*json_data, mod_syslib, device_type, device_id), name));
      PackedFunc load_params = this->models_[name].GetFunction("load_params_contig");
      this->models_[name].status = ModelStatus::EVICTED;
      this->mapLock_.unlock();

      load_params(params_arr);

      this->mapLock_.lock();
      this->modelLocks_[name].unlock();
      this->mapLock_.unlock();

      delete json_data;
      delete params_data;

      sourceLock_.lock();
      modelSource_.erase(name);
      sourceLock_.unlock();

      return [=](){
        ret->set_value();
      }; // nothing happens on gpu so pass a nop
    }, prev, name);
    Task* next = cpu_.addTask(createModel);

    prev->setNextTask(next);

    cpuExecLock_.unlock();

    return ret->get_future();
  }

  std::future<void> Manager::loadToGPUAndInfer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    // lock on executors and enqueue the tasks
    this->execLock_.lock();
    model.status = ModelStatus::IN_USE;

    Task copyToDevice("load_to_device", [=, &model] {
      PackedFunc load = model.GetFunction("load_to_device");
      auto ctx = model.GetFunction("get_contig_context")();

      void* memaddr = load().operator void*();
      ManagedCUDADeviceAPI::Global()->ClaimOwnership(ctx, memaddr, model.name);
      return []() {};
    }, model.name);
    Task* prev = this->load_to_device_.addTask(copyToDevice);

    Task uploadInput("input", [=, &model] {
      PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
      return [](){}; // nothing happens on gpu so pass a nop
    }, prev, model.name);
    Task* next = upload_inputs_.addTask(uploadInput);

    prev->setNextTask(next);
    prev = next;

    Task run("run", [=, &model] {
      PackedFunc run = model.GetFunction("run");
      run();
      return [](){}; // nothing happens on gpu so pass a nop
    }, prev, model.name);
    next = gpu_.addTask(run);

    prev->setNextTask(next);
    prev = next;

    Task getOutput("output", [=, &model] {
      PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);

      return [=, &model](){
        // update model status
        model.status = ModelStatus::READY;
        model.last_use = std::chrono::high_resolution_clock::now();
        //TIMESTAMP("INFERENCE COMPLETE");

        // release model for use
        this->mapLock_.lock();
        this->modelLocks_[model.name].unlock();
        this->mapLock_.unlock();

        ret->set_value();
      };
    }, prev, model.name);
    this->d2h_pcie_.addTask(getOutput);

    prev->setNextTask(next);

    this->execLock_.unlock();

    return ret->get_future();
  }

  std::future<void> Manager::infer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    // lock on executors and enqueue the tasks
    this->execLock_.lock();
    model.status = ModelStatus::IN_USE;

    Task uploadInput("input", [=, &model] {
      PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
      return [](){}; // nothing happens on gpu so pass a nop
    }, model.name);
    Task* prev = upload_inputs_.addTask(uploadInput);

    Task run("run", [=, &model] {
      PackedFunc run = model.GetFunction("run");
      run();
      return [](){}; // nothing happens on gpu so pass a nop
    }, prev, model.name);
    Task* next = gpu_.addTask(run);

    prev->setNextTask(next);
    prev = next;

    Task getOutput("output", [=, &model] {
      PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);

      return [=, &model](){
        // update model status
        model.status = ModelStatus::READY;
        model.last_use = std::chrono::high_resolution_clock::now();
        //TIMESTAMP("INFERENCE COMPLETE");

        // release model for use
        this->mapLock_.lock();
        this->modelLocks_[model.name].unlock();
        this->mapLock_.unlock();

        ret->set_value();
      };
    }, prev, model.name);
    next = this->d2h_pcie_.addTask(getOutput);

    prev->setNextTask(next);

    this->execLock_.unlock();

    return ret->get_future();
  }

}  // namespace runtime
}  // namespace tvm
