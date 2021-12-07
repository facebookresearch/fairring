/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAStream.h>
#include <c10d/NCCLUtils.hpp>
#include <c10d/Store.hpp>
#include <fairring/process_group.h>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace {

int64_t deltaAsUs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
      .count();
}

#define NCCL_CHECK(op)                        \
  {                                           \
    ncclResult_t res = (op);                  \
    if (res != ncclSuccess) {                 \
      throw std::runtime_error("NCCL error"); \
    }                                         \
  }

// We need this extra named namespace inside our unnamed namespace because of
// https://github.com/pybind/pybind11/issues/3289
namespace benchmark_fairring {

class Client {
 public:
  Client(
      size_t machineIdx,
      size_t deviceIdx,
      size_t numMachines,
      size_t numDevicesPerMachine,
      size_t numBuckets,
      size_t bucketSize,
      size_t numEpochs,
      c10::intrusive_ptr<c10d::Store> store,
      bool useNccl,
      std::optional<size_t> parallelism)
      : machineIdx_(machineIdx),
        deviceIdx_(deviceIdx),
        numMachines_(numMachines),
        numDevicesPerMachine_(numDevicesPerMachine),
        numBuckets_(numBuckets),
        bucketSize_(bucketSize),
        numEpochs_(numEpochs),
        store_(std::move(store)) {
    if (useNccl) {
      pg_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
          store_,
          /*rank=*/machineIdx_ * numDevicesPerMachine_ + deviceIdx_,
          /*size=*/numMachines_ * numDevicesPerMachine_,
          c10d::ProcessGroupNCCL::Options::create(
              /*isHighPriorityStream=*/true));
    } else {
      pg_ = c10::make_intrusive<fairring::ProcessGroupFairring>(
          store_,
          /*rank=*/machineIdx_ * numDevicesPerMachine_ + deviceIdx_,
          /*size=*/numMachines_ * numDevicesPerMachine_,
          fairring::ProcessGroupFairring::OptionsFairring::create(
              /*maxMemoryAllocatedInBytes=*/parallelism.value_or(numBuckets_) *
                  ((2 * numMachines_ + 1) * bucketSize_ /
                       numDevicesPerMachine_ / numMachines_ * 4 +
                   numDevicesPerMachine_ * 8),
              /*sliceSizeInBytes=*/bucketSize_ * 4,
              /*isHighPriorityStream=*/true));
    }
  }

  std::vector<int64_t> run() {
    allocateTensors();
    std::vector<int64_t> stats;
    for (size_t epochIdx = 0; epochIdx < numEpochs_; epochIdx += 1) {
      setTensorsToOne();
      {
        auto start = std::chrono::steady_clock::now();
        runOneEpoch();
        auto stop = std::chrono::steady_clock::now();
        stats.push_back(deltaAsUs(start, stop));
      }
      checkTensors();
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return stats;
  }

 private:
  const size_t machineIdx_;
  const size_t deviceIdx_;
  const size_t numMachines_;
  const size_t numDevicesPerMachine_;
  const size_t numBuckets_;
  const size_t bucketSize_;
  const size_t numEpochs_;
  const c10::intrusive_ptr<c10d::Store> store_;
  std::vector<torch::Tensor> buckets_;
  at::Tensor bucketStorage_;
  c10::intrusive_ptr<c10d::ProcessGroup> pg_;

  void allocateTensors() {
    bucketStorage_ = torch::empty(
        {static_cast<int64_t>(numBuckets_), static_cast<int64_t>(bucketSize_)},
        c10::TensorOptions()
            .dtype(c10::kFloat)
            .device(c10::Device(c10::kCUDA, 0)));
    buckets_.reserve(numBuckets_);
    for (size_t bucketIdx = 0; bucketIdx < numBuckets_; bucketIdx += 1) {
      buckets_.push_back(bucketStorage_[bucketIdx]);
    }
  }

  void setTensorsToOne() {
    c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromPool(/*isHighPriority=*/true, /*device=*/0);
    c10::cuda::CUDAStreamGuard g(stream);

    for (size_t bucketIdx = 0; bucketIdx < numBuckets_; bucketIdx += 1) {
      // at::arange_out(buckets_[bucketIdx], buckets_[bucketIdx].numel());
      // buckets_[bucketIdx].fill_(1);
    }
    bucketStorage_.fill_(1);

    stream.synchronize();
  }

  void runOneEpoch() {
    c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromPool(/*isHighPriority=*/true, /*device=*/0);
    c10::cuda::CUDAStreamGuard g(stream);

    std::vector<c10::intrusive_ptr<c10::ivalue::Future>> futures;
    for (size_t bucketIdx = 0; bucketIdx < numBuckets_; bucketIdx += 1) {
      std::vector<at::Tensor> data = {buckets_[bucketIdx]};
      futures.push_back(pg_->allreduce(data)->getFuture());
    }

    for (const auto& future : futures) {
      // future->wait(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::minutes(1)));
      future->wait();
    }

    stream.synchronize();
  }

  void checkTensors() {
    for (size_t bucketIdx = 0; bucketIdx < numBuckets_; bucketIdx += 1) {
      // at::Tensor expected = at::arange(
      //                           buckets_[bucketIdx].numel(),
      //                           c10::TensorOptions()
      //                               .dtype(c10::kFloat)
      //                               .device(c10::Device(c10::kCUDA, 0))) *
      //     static_cast<int64_t>(numMachines_ * numDevicesPerMachine_);
      at::Tensor expected =
          torch::full(
              {1},
              static_cast<float>(numMachines_ * numDevicesPerMachine_),
              c10::TensorOptions()
                  .dtype(c10::kFloat)
                  .device(c10::Device(c10::kCUDA, 0)))
              .expand({buckets_[bucketIdx].numel()});

      // if (!buckets_[bucketIdx].allclose(expected)) {
      //   throw std::runtime_error("Bad result");
      // }
      at::Tensor closeness =
          buckets_[bucketIdx].isclose(expected).logical_not();
      at::Tensor nonCloseIndices = closeness.nonzero();
      if (nonCloseIndices.size(0) > 0) {
        LOG(ERROR) << "In bucket " << bucketIdx << " which starts at 0x"
                   << std::hex
                   << reinterpret_cast<uintptr_t>(
                          buckets_[bucketIdx].data_ptr())
                   << std::dec << " found non-close value at index "
                   << nonCloseIndices[0].item<int64_t>() << " which has value "
                   << buckets_[bucketIdx][nonCloseIndices[0].item<int64_t>()]
                          .item<float>()
                   << " instead of "
                   << expected[nonCloseIndices[0].item<int64_t>()].item<float>()
                   << " and there are " << nonCloseIndices.size(0)
                   << " non-close values in total ";
      }
    }
  }
};

} // namespace benchmark_fairring
} // namespace

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PYBIND11_MODULE(benchmark_fairring, module) {
  shared_ptr_class_<benchmark_fairring::Client> client(module, "Client");

  client.def(
      py::init<
          size_t,
          size_t,
          size_t,
          size_t,
          size_t,
          size_t,
          size_t,
          const c10::intrusive_ptr<c10d::Store>&,
          bool,
          std::optional<size_t>>(),
      py::arg("machine_idx"),
      py::arg("device_idx"),
      py::arg("num_machines"),
      py::arg("num_devices_per_machine"),
      py::arg("num_buckets"),
      py::arg("bucket_size"),
      py::arg("num_epochs"),
      py::arg("store"),
      py::arg("use_nccl"),
      py::arg("parallelism"));
  client.def(
      "run",
      &benchmark_fairring::Client::run,
      py::call_guard<py::gil_scoped_release>());
}
