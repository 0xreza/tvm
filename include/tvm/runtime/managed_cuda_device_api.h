/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.h
 * \brief GPU specific API
 */
#ifndef TVM_RUNTIME_CUDA_MANAGED_H_
#define TVM_RUNTIME_CUDA_MANAGED_H_
#include <tvm/runtime/device_api.h>

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <cuda_runtime.h>
#include <tvm/runtime/cuda_common.h>
#include <string>
#include <iostream>
#include <sstream>

namespace tvm {
namespace runtime {

static size_t total_copied_m = 0;

/** Simple interface for implementing custom memory managers */
class CUDAMemoryManager {
public:
  virtual void* alloc(int device, size_t nbytes, size_t alignment) = 0;
  virtual void free(int device, void* ptr) = 0;
};

/** Memory manager that directly proxies cudaMalloc */
class CUDAMallocMemoryManager : public CUDAMemoryManager {
public:

  void* alloc(int device, size_t nbytes, size_t alignment) {
      CUDA_CALL(cudaSetDevice(device));
      CHECK_EQ(256 % alignment, 0U)
          << "CUDA space is aligned at 256 bytes";
      void *ret;
      CUDA_CALL(cudaMalloc(&ret, nbytes));
      return ret;

  }

  void free(int device, void* ptr) {
      CUDA_CALL(cudaSetDevice(device));
      CUDA_CALL(cudaFree(ptr));
  }

};

class ManagedCUDADeviceAPI final : public DeviceAPI {
private:
  CUDAMemoryManager* dataspacemanager = new CUDAMallocMemoryManager();
  CUDAMemoryManager* workspacemanager = dataspacemanager;

public:

  void SetDataspaceManager(CUDAMemoryManager* manager) {
    dataspacemanager = manager;
  }

  void SetWorkspaceManager(CUDAMemoryManager* manager) {
    workspacemanager = manager;
  }

  void SetDevice(TVMContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU GET_ATTR " << kind << std::endl;
    switch (kind) {
      case kExist:
        value = (
            cudaDeviceGetAttribute(
                &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id)
            == cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        cudaDeviceProp props;
        CUDA_CALL(cudaGetDeviceProperties(&props, ctx.device_id));
        *rv = std::string(props.name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMultiProcessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[0], cudaDevAttrMaxBlockDimX, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[1], cudaDevAttrMaxBlockDimY, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[2], cudaDevAttrMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
    }
    *rv = value;
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint,
                       bool workspace = false) final {
    if (workspace) {
      return workspacemanager->alloc(ctx.device_id, nbytes, alignment);
    } else {
      return dataspacemanager->alloc(ctx.device_id, nbytes, alignment);
    }
  }

  void FreeDataSpace(TVMContext ctx, void* ptr, bool workspace = false) final {
    if (workspace) {
      workspacemanager->free(ctx.device_id, ptr);
    } else {
      dataspacemanager->free(ctx.device_id, ptr);
    }
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    // // std::cout << LOG_PREFIX() << "CopyDataFromTo " << size << std::endl;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(ManagedCUDAThreadEntry::ThreadLocal()->stream);
    // // IGNORE passed in stream?
    //cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLGPU) {
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM GPU_GPU_COPY " << size << std::endl;
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM GPU_OGPU_COPY " << size << std::endl;
        CUDA_CALL(cudaMemcpyPeerAsync(to, ctx_to.device_id,
                            from, ctx_from.device_id,
                            size, cu_stream));
      }
    } else if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLCPU) {
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM GPU_CPU_COPY " << size << std::endl;
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLGPU) {
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM CPU_GPU_COPY " << size << std::endl;
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }

    // // std::cout << LOG_PREFIX() << "CopyDataFromTo exit" << std::endl;
  }

  TVMStreamHandle CreateStream(TVMContext ctx) {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t retval;
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM CREATE_STREAM " << retval << std::endl;
    CUDA_CALL(cudaStreamCreate(&retval));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(TVMContext ctx, TVMStreamHandle stream) {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM FREE_STREAM " << stream << std::endl;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(TVMContext ctx, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM STREAM_FROM_TO " << event_src << " " << event_dst << std::endl;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU SETDEVICE" << std::endl;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM SYNC_STREAM " << stream << std::endl;
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    ManagedCUDAThreadEntry::ThreadLocal()
        ->stream = static_cast<cudaStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    // // std::cout << "Allocworkspace " << size << std::endl;
    void* address = ManagedCUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM ALLOC_WORK " << size << " " << address << std::endl;
    return address;
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    // std::cout << "\t" << LOG_PREFIX() << "RSC USAGE: GPU MEM FREE_WORK " << data << std::endl;
    ManagedCUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<ManagedCUDADeviceAPI>& Global() {
    static std::shared_ptr<ManagedCUDADeviceAPI> inst =
        std::make_shared<ManagedCUDADeviceAPI>();
    return inst;
  }

 private:

  static void GPUCopy(const void* from,
                      void* to,
                      size_t size,
                      cudaMemcpyKind kind,
                      cudaStream_t stream) {
    // if (stream != 0) {
    // // std::cout << LOG_PREFIX() << " copying " << size << " to " << stream << std::endl;
      REGULAR_LOG("copying " << size << " to stream " << stream << " --- total " << (total_copied_m += size));
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));

    // // std::cout << LOG_PREFIX() << " FINISHED copying " << size << " to " << stream << std::endl;
    // } else {
    //   CUDA_CALL(cudaMemcpy(to, from, size, kind));
    // }
  }
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CUDA_MANAGED_H_