/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */

 #include <tvm/runtime/executor.h>
 #include <tvm/runtime/cuda_common.h>
 #include <chrono>
 #include <pthread.h>
 #include <thread>

 namespace tvm {
 namespace runtime {

   std::mutex outLock;

   Executor::Executor(bool isLast, int threads) : keepGoing_(true),
                                                isLast_(isLast),
                                                nThreads_(threads),
                                                tp_(threads),
                                                runner_(&Executor::run, this) {
                                                  this->runner_.detach();
                                                }

  std::atomic<int> Executor::global_exec_count(0);

  void set_core(unsigned core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
  }

   void Executor::run() {
     if (nThreads_ > 1) return; // multithreaded executor, let thread pool handle running
     int id = (std::thread::hardware_concurrency() - global_exec_count.fetch_add(1)) % std::thread::hardware_concurrency();
     id = (id >= 0) ? id : id + std::thread::hardware_concurrency();

     set_core(static_cast<unsigned>(id));

     // events used for timing each task
     cudaEvent_t start, end;
     CUDA_CALL(cudaEventCreate(&start));
     CUDA_CALL(cudaEventCreate(&end));

     while (keepGoing_.load()) {
       taskMutex_.lock();
       if (tasks_.size() == 0) {
         taskMutex_.unlock();
         continue;
       }
       Task& t = tasks_.front();
       taskMutex_.unlock();

       t.handleDep.lock();
       if (t.previousTask != nullptr) {
         // wait for previous task to start
         while (!t.previousTask->started.load()) {}

         // make this stream wait on previous executor stream contents
         CUDA_CALL(cudaStreamWaitEvent(ManagedCUDAThreadEntry::ThreadLocal()->stream, t.previousTask->sync, 0));

         // mark the dependency as done so that other executor can move to the next task
         t.previousTask->nextHasCecked.store(true);
       }
       t.handleDep.unlock();

       // event that will record all work queued on gpu stream
       CUDA_CALL(cudaEventCreateWithFlags(&t.sync, cudaEventBlockingSync | cudaEventDisableTiming));

       // actually queue the operation and time it
       CUDA_CALL(cudaEventRecord(start, ManagedCUDAThreadEntry::ThreadLocal()->stream));

       auto postOp = t.operation();

       //LOCKED_LOG("TASK_QUEUED " + t.task_name_, t.modelname);

       // record content of stream to sync on in the next executor's stream
       CUDA_CALL(cudaEventRecord(t.sync, ManagedCUDAThreadEntry::ThreadLocal()->stream));
       CUDA_CALL(cudaEventRecord(end, ManagedCUDAThreadEntry::ThreadLocal()->stream));

       // notify the next task that it can at least queue the next operation
       t.started.store(true);

       // wait for operation to finish
       CUDA_CALL(cudaStreamSynchronize(ManagedCUDAThreadEntry::ThreadLocal()->stream));

       postOp();

       // calculate runtime
       float duration = 0.0f;
       CUDA_CALL(cudaEventElapsedTime(&duration, start, end));

       //LOCKED_LOG_TIME(t.modelname, "TASK_DUR " + t.task_name_, duration);

       // if the next task has already seen that this has started, it can be deleted
       // if it hasn't, remove the dependency as the task has already completed
       if (t.nextTask != nullptr) {
         t.nextTask->handleDep.lock();
         if (!t.nextHasCecked.load()) {
           t.nextTask->previousTask = nullptr;
         }
         t.nextTask->handleDep.unlock();
       }

       // if (!isLast_) {
       //   while (!t.nextHasCecked.load()) {}
       // }

       // CUDA_CALL(cudaEventDestroy(t.sync));

       taskMutex_.lock();
       tasks_.pop();
       taskMutex_.unlock();
     }
   }

   Task* Executor::addTask(Task& task) {
     if (nThreads_ == 1) {
       // std::cout << "ok but what about this\n";
       taskMutex_.lock();
       tasks_.push(std::move(task));
       Task* ret = &(tasks_.back());
       taskMutex_.unlock();
       return ret;
     } else {
       // std::cout << "are we getting here?\n";
       Task* t = new Task(std::move(task));
       std::function<void(void)> exec = [=](){
         // events used for timing each task
         cudaEvent_t start, end;
         CUDA_CALL(cudaEventCreate(&start));
         CUDA_CALL(cudaEventCreate(&end));
         t->handleDep.lock();

         if (t->previousTask != nullptr) {
           // wait for previous task to start
           while (!t->previousTask->started.load()) {}

           // make this stream wait on previous executor stream contents
           CUDA_CALL(cudaStreamWaitEvent(ManagedCUDAThreadEntry::ThreadLocal()->stream, t->previousTask->sync, 0));

           // mark the dependency as done so that other executor can move to the next task
           t->previousTask->nextHasCecked.store(true);
         }
         t->handleDep.unlock();

         // event that will record all work queued on gpu stream
         CUDA_CALL(cudaEventCreateWithFlags(&t->sync, cudaEventBlockingSync | cudaEventDisableTiming));

         // actually queue the operation and time it
         CUDA_CALL(cudaEventRecord(start, ManagedCUDAThreadEntry::ThreadLocal()->stream));

         auto postOp = t->operation();

         //LOCKED_LOG("TASK_QUEUED " + t->task_name_, t->modelname);

         // record content of stream to sync on in the next executor's stream
         CUDA_CALL(cudaEventRecord(t->sync, ManagedCUDAThreadEntry::ThreadLocal()->stream));
         CUDA_CALL(cudaEventRecord(end, ManagedCUDAThreadEntry::ThreadLocal()->stream));

         // notify the next task that it can at least queue the next operation
         t->started.store(true);

         // wait for operation to finish
         CUDA_CALL(cudaStreamSynchronize(ManagedCUDAThreadEntry::ThreadLocal()->stream));

         postOp();

         // calculate runtime
         float duration = 0.0f;
         CUDA_CALL(cudaEventElapsedTime(&duration, start, end));

         //LOCKED_LOG_TIME(t->modelname, "TASK_DUR " + t->task_name_, duration);

         // if the next task has already seen that this has started, it can be deleted
         // if it hasn't, remove the dependency as the task has already completed
         if (t->nextTask != nullptr) {
           t->nextTask->handleDep.lock();
           if (!t->nextHasCecked.load()) {
             t->nextTask->previousTask = nullptr;
           }
           t->nextTask->handleDep.unlock();
         }

         delete t;
       };
       tp_.push(exec);

       return t;
     }
   }

 }  // namespace runtime
 }  // namespace tvm
