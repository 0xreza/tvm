/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_COMMON_H_
#define TVM_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include <tvm/runtime/workspace_pool.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <cstring>
#include <sstream>

namespace tvm {
namespace runtime {

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

inline std::string now() {
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();

  typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8>
  >::type> Days; /* UTC: +8:00 */

  Days days = std::chrono::duration_cast<Days>(duration);
      duration -= days;
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
      duration -= hours;
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
      duration -= minutes;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
      duration -= seconds;
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
      duration -= milliseconds;
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
      duration -= microseconds;
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

  std::stringstream ss;
  ss << hours.count() << ":"
            << minutes.count() << ":"
            << seconds.count() << "."
            << milliseconds.count() << " "
            << microseconds.count() << " "
            << nanoseconds.count();
  return ss.str();
}


// Dummy parameter-pack expander
template <class T>
void expand(std::initializer_list<T>) {}

// Fun
template <class Fun, class... Args>
typename std::result_of<Fun&&(Args&&...)>::type
call(Fun&& f, Args&&... args) {

    // Print all parameters
    expand({(std::cout << args << ' ', 0)...});

    // Forward the call
    return std::forward<Fun>(f)(std::forward<Args>(args)...);
}

#define LOG_PREFIX() now() << " " << __FILENAME__ << ":" << __LINE__ << " \t"

// #define LOGGINGENABLED

#ifdef LOGGINGENABLED

    #define REGULAR_LOG(x) std::cout << LOG_PREFIX() << x << std::endl;

    #define CUDA_DRIVER_CALL(x)                                             \
      {                                                                     \
        std::cout << now() << "  " << __FILENAME__ << ":" << __LINE__ << " \t" << #x << std::endl; \
        CUresult result = x;                                                \
        if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
          const char *msg;                                                  \
          cuGetErrorName(result, &msg);                                     \
          LOG(FATAL)                                                        \
              << "CUDAError: " #x " failed with error: " << msg;            \
        }                                                                   \
      }

    #define CUDA_LOG(f) \
      std::cout << now() << "  " << __FILENAME__ << ":" << __LINE__ << " \t" << #f << std::endl; \
      f \


    #define CUDA_CALL(func)                                            \
      {                                                           \
        std::cout << now() << "  " << __FILENAME__ << ":" << __LINE__ << " \t" << #func << std::endl; \
        cudaError_t e = (func);                                        \
        CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
            << "CUDA: " << cudaGetErrorString(e);                      \
      }


    #define CUDA_CALLM(func, ...)                                            \
      {                                                                \
        std::cout << now() << "  " << __FILENAME__ << ":" << __LINE__ << " \t" << #func \
        cudaError_t e = (func);                                        \
        CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
            << "CUDA: " << cudaGetErrorString(e);                      \
      }

#else

    #define REGULAR_LOG(x)

    #define CUDA_DRIVER_CALL(x)                                             \
      {                                                                     \
        CUresult result = x;                                                \
        if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
          const char *msg;                                                  \
          cuGetErrorName(result, &msg);                                     \
          LOG(FATAL)                                                        \
              << "CUDAError: " #x " failed with error: " << msg;            \
        }                                                                   \
      }

    #define CUDA_LOG(f) f


    #define CUDA_CALL(func)                                            \
      {                                                           \
        cudaError_t e = (func);                                        \
        CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
            << "CUDA: " << cudaGetErrorString(e);                      \
      }


    #define CUDA_CALLM(func, ...)                                            \
      {                                                                \
        cudaError_t e = (func);                                        \
        CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
            << "CUDA: " << cudaGetErrorString(e);                      \
      }
#endif


/*! \brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};

/*! \brief Thread local workspace */
class ManagedCUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  ManagedCUDAThreadEntry();
  // get the threadlocal workspace
  static ManagedCUDAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CUDA_CUDA_COMMON_H_
