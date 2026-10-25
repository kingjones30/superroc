// Persistent FIFO thread pool for gufo-quant (v0.6).
//
// v0.5 spawned (and joined) up to nthread std::threads per tensor -- 866 join
// rounds for the 27B -- and serialized on small tensors (a tensor with fewer
// rows than threads idles the rest). This pool is created once for the run:
// workers pull fixed chunk tasks from one FIFO queue, so big tensors split
// across every worker and small tensors steal across tensor boundaries. Task
// boundaries are a fixed function of tensor shape (see kChunkElems in the
// caller), never of thread count or scheduling, and every task writes a
// disjoint output range -- so bytes are identical for any worker count.
//
// Exceptions: a task that throws records the first exception and sets the
// error flag; the pool keeps draining (fast -- remaining tasks are dropped
// only by the caller's own checks) and the caller rethrows it in its own
// thread after Shutdown (v0.9: gufo-quant catches it and exits 2; it no longer
// lets the exception escape main). A task may legitimately run after a
// sibling threw; the pool drains those, it does not abandon the queue.
//
// v0.8 (RV-LQP2 F2): a throwing task must WAKE the caller's waiter, which may
// sleep on a different condition variable than the pool's own done_cv_ (the
// quantizer's writer waits on its pipeline cv and only polls pool.error() on
// wakeup -- a throw skipped the only notify and hung the run forever). The
// caller registers SetErrorCallback; the worker invokes it once per throwing
// task, AFTER releasing the pool mutex (the callback must not call back into
// the pool). A null callback (the default) leaves the caller to poll
// pool.error() as before.

#ifndef GUFO_QUANT_POOL_HPP_
#define GUFO_QUANT_POOL_HPP_

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gufo::quant_tool {

class ThreadPool {
 public:
  explicit ThreadPool(int nworkers) : stop_(false), active_(0) {
    if (nworkers < 1) nworkers = 1;
    for (int i = 0; i < nworkers; ++i) {
      workers_.emplace_back([this]() {
        for (;;) {
          std::function<void()> fn;
          {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]() { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            fn = std::move(queue_.front());
            queue_.erase(queue_.begin());
            ++active_;
          }
          try {
            fn();
          } catch (...) {
            std::function<void()> cb;
            {
              std::lock_guard<std::mutex> lk(mu_);
              if (!first_exception_) first_exception_ = std::current_exception();
              error_ = true;
              cb = error_callback_;
            }
            // Outside the pool mutex: wake the caller's waiter (F2). The
            // callback must not call back into this pool.
            if (cb) cb();
          }
          {
            std::lock_guard<std::mutex> lk(mu_);
            --active_;
            if (queue_.empty() && active_ == 0) done_cv_.notify_all();
          }
        }
      });
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // No more submissions after Shutdown; joins every worker. Idempotent.
  void Shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_) return;
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
    workers_.clear();
  }

  ~ThreadPool() { Shutdown(); }

  void Submit(std::function<void()> fn) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(std::move(fn));
    }
    cv_.notify_one();
  }

  // v0.8 (F2): invoked once per throwing task, outside the pool mutex, so a
  // waiter on the caller's own condition variable wakes. Set before
  // submitting; the callback must not call back into this pool.
  void SetErrorCallback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lk(mu_);
    error_callback_ = std::move(cb);
  }

  // Blocks until every submitted task finished (submissions from other
  // threads may still be in flight; callers needing a hard barrier must stop
  // submitting first). Returns false when a task threw.
  bool WaitIdle() {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this]() { return queue_.empty() && active_ == 0; });
    return !error_;
  }

  bool error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
  }

  // Rethrows the first task exception (no-op when none). Call in the main
  // thread after the work is known done so the exit path matches a direct
  // throw from the same code.
  void RethrowIfError() {
    std::exception_ptr e;
    {
      std::lock_guard<std::mutex> lk(mu_);
      e = first_exception_;
    }
    if (e) std::rethrow_exception(e);
  }

 private:
  std::vector<std::thread> workers_;
  std::vector<std::function<void()>> queue_;  // FIFO (pop front); tiny
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  bool stop_;
  int active_;
  bool error_ = false;
  std::exception_ptr first_exception_;
  std::function<void()> error_callback_;  // under mu_; invoked without it
};

}  // namespace gufo::quant_tool

#endif  // GUFO_QUANT_POOL_HPP_
