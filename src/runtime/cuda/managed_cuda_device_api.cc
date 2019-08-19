/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/managed_cuda_device_api.h>

namespace tvm {
namespace runtime {

typedef dmlc::ThreadLocalStore<ManagedCUDAThreadEntry> ManagedCUDAThreadStore;

ManagedCUDAThreadEntry::ManagedCUDAThreadEntry()
    : pool(kDLGPU, ManagedCUDADeviceAPI::Global()) {
}

ManagedCUDAThreadEntry* ManagedCUDAThreadEntry::ThreadLocal() {
  // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU THREAD ENTRY" << std::endl;
  return ManagedCUDAThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.gpu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = ManagedCUDADeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
