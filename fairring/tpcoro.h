/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <coroutine>

#include <tensorpipe/tensorpipe.h>

#include <fairring/utils.h>

namespace tpcoro {

class Task {
 private:
  struct SharedState {
    std::mutex mutex_;
    bool done_ = false;
    std::exception_ptr error_ = nullptr;
    std::coroutine_handle<> handle_ = nullptr;
  };

  class Promise {
   public:
    Promise() : state_(std::make_shared<SharedState>()) {}

    Task get_return_object() {
      return Task(state_);
    }

    std::suspend_never initial_suspend() {
      return {};
    }

    std::suspend_never final_suspend() noexcept {
      return {};
    }

    void return_void() {
      std::coroutine_handle<> handle;
      {
        std::unique_lock<std::mutex> lock(state_->mutex_);
        state_->done_ = true;
        std::swap(state_->handle_, handle);
      }
      if (handle != nullptr) {
        handle.resume();
      }
    }

    void unhandled_exception() {
      std::coroutine_handle<> handle;
      {
        std::unique_lock<std::mutex> lock(state_->mutex_);
        state_->done_ = true;
        state_->error_ = std::current_exception();
        std::swap(state_->handle_, handle);
      }
      if (handle != nullptr) {
        handle.resume();
      }
    }

   private:
    const std::shared_ptr<SharedState> state_;
  };

  Task(std::shared_ptr<SharedState> state) : state_(std::move(state)) {}

  const std::shared_ptr<SharedState> state_;

 public:
  using promise_type = Promise;

  bool await_ready() {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    return state_->done_;
  }

  bool await_suspend(std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    MY_CHECK(state_->handle_ == nullptr);
    if (state_->done_) {
      return false;
    } else {
      state_->handle_ = std::move(handle);
      return true;
    }
  }

  void await_resume() {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    MY_CHECK(state_->done_);
    if (state_->error_) {
      std::rethrow_exception(state_->error_);
    }
  }
};

class Parallel {
 public:
  Parallel(std::vector<Task> tasks) {
    numRemaining_ = tasks.size();
    for (Task& t : tasks) {
      runOne(std::move(t));
    }
  }

  Parallel() = delete;
  DELETE_COPY_MOVE_CONSTRUCTORS(Parallel)

  bool await_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    return numRemaining_ == 0;
  }

  bool await_suspend(std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(handle_ == nullptr);
    if (numRemaining_ == 0) {
      return false;
    } else {
      handle_ = std::move(handle);
      return true;
    }
  }

  void await_resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(numRemaining_ == 0);
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  ~Parallel() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (numRemaining_ != 0) {
      std::cerr << "Destroying an unawaited Parallel" << std::endl;
      std::terminate();
    }
  }

 private:
  std::mutex mutex_;
  size_t numRemaining_ = 0;
  std::exception_ptr error_ = nullptr;
  std::coroutine_handle<> handle_ = nullptr;

  Task runOne(Task task) {
    try {
      co_await task;
      markOneCompleted(nullptr);
    } catch (...) {
      markOneCompleted(std::current_exception());
    }
  }

  void markOneCompleted(std::exception_ptr error) {
    std::coroutine_handle<> handle;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      numRemaining_ -= 1;
      if (error != nullptr && error_ == nullptr) {
        error_ = error;
      }
      if (numRemaining_ == 0) {
        std::swap(handle_, handle);
      }
    }
    if (handle != nullptr) {
      handle.resume();
    }
  }
};

class TpError : public std::exception {
 public:
  TpError(const tensorpipe::Error& error) : reason_(error.what()) {}

  const char* what() const noexcept override {
    return reason_.c_str();
  }

 private:
  std::string reason_;
};

class TpWrite {
 public:
  TpWrite(tensorpipe::Pipe& pipe, tensorpipe::Message message) {
    pipe.write(std::move(message), [this](const tensorpipe::Error& error) {
      std::coroutine_handle<> handle;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        if (error) {
          error_ = error;
        }
        std::swap(handle_, handle);
      }
      if (handle != nullptr) {
        handle.resume();
      }
    });
  }

  TpWrite() = delete;
  DELETE_COPY_MOVE_CONSTRUCTORS(TpWrite)

  bool await_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    return done_;
  }

  bool await_suspend(std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(handle_ == nullptr);
    if (done_) {
      return false;
    } else {
      handle_ = std::move(handle);
      return true;
    }
  }

  void await_resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(done_);
    if (error_) {
      throw TpError(error_);
    }
  }

  ~TpWrite() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!done_) {
      std::cerr << "Destroying an unawaited TpWrite" << std::endl;
      std::terminate();
    }
  }

 private:
  std::mutex mutex_;
  bool done_ = false;
  tensorpipe::Error error_ = tensorpipe::Error::kSuccess;
  std::coroutine_handle<> handle_ = nullptr;
};

class TpReadDescriptor {
 public:
  TpReadDescriptor(tensorpipe::Pipe& pipe) {
    pipe.readDescriptor(
        [this](
            const tensorpipe::Error& error, tensorpipe::Descriptor descriptor) {
          std::coroutine_handle<> handle;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            done_ = true;
            if (error) {
              error_ = error;
            } else {
              result_ = std::move(descriptor);
            }
            std::swap(handle_, handle);
          }
          if (handle != nullptr) {
            handle.resume();
          }
        });
  }

  TpReadDescriptor() = delete;
  DELETE_COPY_MOVE_CONSTRUCTORS(TpReadDescriptor)

  bool await_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    return done_;
  }

  bool await_suspend(std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(handle_ == nullptr);
    if (done_) {
      return false;
    } else {
      handle_ = std::move(handle);
      return true;
    }
  }

  void await_resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(done_);
    if (error_) {
      throw TpError(error_);
    }
  }

  // Ideally we'd make await_resume return the tensorpipe::Descriptor, but this
  // triggers a GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101133
  tensorpipe::Descriptor&& getDescriptor() && {
    return std::move(result_);
  }

  ~TpReadDescriptor() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!done_) {
      std::cerr << "Destroying an unawaited TpReadDescriptor" << std::endl;
      std::terminate();
    }
  }

 private:
  std::mutex mutex_;
  bool done_ = false;
  tensorpipe::Descriptor result_;
  tensorpipe::Error error_ = tensorpipe::Error::kSuccess;
  std::coroutine_handle<> handle_ = nullptr;
};

class TpRead {
 public:
  TpRead(tensorpipe::Pipe& pipe, tensorpipe::Allocation allocation) {
    pipe.read(std::move(allocation), [this](const tensorpipe::Error& error) {
      std::coroutine_handle<> handle;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        if (error) {
          error_ = error;
        }
        std::swap(handle_, handle);
      }
      if (handle != nullptr) {
        handle.resume();
      }
    });
  }

  TpRead() = delete;
  DELETE_COPY_MOVE_CONSTRUCTORS(TpRead)

  bool await_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    return done_;
  }

  bool await_suspend(std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(handle_ == nullptr);
    if (done_) {
      return false;
    } else {
      handle_ = std::move(handle);
      return true;
    }
  }

  void await_resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    MY_CHECK(done_);
    if (error_) {
      throw TpError(error_);
    }
  }

  ~TpRead() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!done_) {
      std::cerr << "Destroying an unawaited TpRead" << std::endl;
      std::terminate();
    }
  }

 private:
  std::mutex mutex_;
  bool done_ = false;
  tensorpipe::Error error_ = tensorpipe::Error::kSuccess;
  std::coroutine_handle<> handle_ = nullptr;
};

class Sequencer {
 public:
  Sequencer() {}

  class TurnToken;

  class Awaitable {
   public:
    DELETE_COPY_MOVE_CONSTRUCTORS(Awaitable);

    bool await_ready() {
      return false;
    }

    bool await_suspend(std::coroutine_handle<> handle) {
      return sequencer_.waitForTurn(seqNum_, std::move(handle));
    }

    TurnToken await_resume() {
      return TurnToken(sequencer_, seqNum_);
    }

   private:
    Awaitable(Sequencer& sequencer, size_t seqNum)
        : sequencer_(sequencer), seqNum_(seqNum) {}

    Sequencer& sequencer_;
    const size_t seqNum_;

    friend Sequencer;
  };

  class TurnToken {
   public:
    DELETE_COPY_MOVE_CONSTRUCTORS(TurnToken);

    void release() {
      sequencer_.finishTurn(seqNum_);
      released_ = true;
    }

    ~TurnToken() {
      if (!released_) {
        release();
      }
    }

   private:
    TurnToken(Sequencer& sequencer, size_t seqNum)
        : sequencer_(sequencer), seqNum_(seqNum) {}

    Sequencer& sequencer_;
    const size_t seqNum_;
    bool released_ = false;

    friend Awaitable;
  };

  Awaitable waitForTurn(size_t seqNum) {
    return Awaitable(*this, seqNum);
  }

 private:
  std::mutex mutex_;
  size_t nextTurn_ = 0;
  std::deque<std::coroutine_handle<>> handles_;

  bool waitForTurn(size_t seqNum, std::coroutine_handle<> handle) {
    std::unique_lock<std::mutex> lock(mutex_);

    MY_CHECK(handle != nullptr);
    if (seqNum == nextTurn_) {
      return false;
    } else {
      size_t index = seqNum - nextTurn_;
      handles_.resize(std::max(handles_.size(), index + 1));
      MY_CHECK(handles_[index] == nullptr);
      handles_[index] = handle;
      return true;
    }
  }

  void finishTurn(size_t seqNum) {
    std::unique_lock<std::mutex> lock(mutex_);

    MY_CHECK(nextTurn_ == seqNum);
    nextTurn_ += 1;
    if (handles_.size() > 0) {
      handles_.pop_front();
    }

    if (handles_.size() > 0 && handles_[0] != nullptr) {
      std::coroutine_handle<> handle;
      std::swap(handle, handles_[0]);
      lock.unlock();
      handle.resume();
    }
  }
};

// Using lambdas as coroutines is error-prone. A lambda consists of an object
// (with some fields, corresponding to the captures) which is callable. Invoking
// it is equivalent to calling its operator() method. Invoking a method is the
// same as invoking a "free" function with a *pointer/reference* to the object
// as an implicit argument. When invoking a coroutine the compiler will allocate
// its state and copy/move the *arguments* there. To put it all together, when
// invoking a lambda coroutine, the body of the function will only store a
// reference to the capture, which could easily become dangling. Oftentimes a
// lambda is defined inline and thus created as a temporary, and will thus be
// destroyed at the end of the expression if it's not stored elsewhere.
// This is a known problem:
// https://stackoverflow.com/questions/60592174/lambda-lifetime-explanation-for-c20-coroutines
// And Folly has a solution for it called co_invoke.
// https://github.com/facebook/folly/blob/5bbfb175cb8fc7edab442f06105d4681654732e9/folly/experimental/coro/Invoke.h#L25
// Our solution here tries to mimic Folly's one (albeit with less complexity),
// and it works by making co_invoke itself a coroutine, so that passing a lambda
// to it ends up storing the lambda's object in the state of co_invoke. Thus the
// lifetime of the lambda will be same as that of the co_invoke coroutine. Since
// then co_invoke calls and co_awaits the lambda we'll be sure that its lifetime
// will exceed the lambda's body's lifetime.
template <typename TFn, typename... TArgs>
inline Task co_invoke(TFn fn, TArgs&&... args) {
  co_await fn(std::forward<TArgs>(args)...);
}

template <typename TFn>
inline std::vector<Task> forIdxInRange(size_t rangeSize, TFn fn) {
  std::vector<Task> tasks;
  tasks.reserve(rangeSize);
  for (const auto idx : c10::irange(rangeSize)) {
    tasks.push_back(co_invoke(fn, idx));
  }
  return tasks;
}

class CoroTracker {
 public:
  class Beacon {
   public:
    DELETE_COPY_MOVE_CONSTRUCTORS(Beacon);

    void release() {
      tracker_.extinguishBeacon();
      released_ = true;
    }

    ~Beacon() {
      if (!released_) {
        release();
      }
    }

   private:
    Beacon(CoroTracker& tracker) : tracker_(tracker) {}

    CoroTracker& tracker_;
    bool released_ = false;

    friend CoroTracker;
  };

  Beacon startTrackingMe() {
    std::unique_lock<std::mutex> lock(mutex_);
    count_ += 1;
    return Beacon(*this);
  }

  void waitForAllCoros() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return count_ == 0; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t count_ = 0;

  void extinguishBeacon() {
    std::unique_lock<std::mutex> lock(mutex_);
    count_ -= 1;
    cv_.notify_all();
  }
};

} // namespace tpcoro
