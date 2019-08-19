/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */

 #include <tvm/runtime/executor.h>
 #include <tvm/runtime/cuda_common.h>
 #include <chrono>

 namespace tvm {
 namespace runtime {

   Executor::Executor(bool sync, bool isLast) : keepGoing_(true),
                                                syncAfterOp_(sync),
                                                isLast_(isLast),
                                                runner_(&Executor::run, this) {
                                                  this->runner_.detach();
                                                }


   void Executor::run() {
     while (keepGoing_.load()) {
       taskMutex_.lock();
       if (tasks_.size() == 0) {
         taskMutex_.unlock();
         continue;
       }
       Task& t = tasks_.front();
       taskMutex_.unlock();

       if (t.previousTask != nullptr) {
         // wait for previous task to be complete
         while (t.previousTask->done.load() == false) {}

         // mark the dependency as done so that other executor
         t.previousTask->nextHasCecked.store(true);
       }

       // actually do the task
       t.operation();

       LOCKED_LOG("TASK_END", t.modelname);

       // mark this task as complete
       t.done.store(true);

       // wait until the next task has seen that this is done before dequeing
       // to be deleted, if necessary
       if (!isLast_) {
         while (!t.nextHasCecked.load()) {}
       }

       taskMutex_.lock();
       tasks_.pop();
       taskMutex_.unlock();
     }
   }

   Task* Executor::addTask(Task& t) {
     taskMutex_.lock();
     tasks_.push(std::move(t));
     Task* ret = &(tasks_.back());
     taskMutex_.unlock();
     return ret;
   }

 }  // namespace runtime
 }  // namespace tvm
