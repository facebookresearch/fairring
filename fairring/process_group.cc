/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fairring/process_group.h>

#include <set>

#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroupNCCL.hpp>

#include <fairring/machine.h>

namespace {

template <typename TSub, typename TSuper>
c10::intrusive_ptr<TSub> intrusive_cast(c10::intrusive_ptr<TSuper> ptr) {
  TSuper* rawPtr = ptr.release();
  TSub* castRawPtr = dynamic_cast<TSub*>(rawPtr);
  if (castRawPtr == nullptr) {
    ptr = c10::intrusive_ptr<TSuper>::reclaim(rawPtr);
    return c10::intrusive_ptr<TSub>();
  }
  return c10::intrusive_ptr<TSub>::reclaim(castRawPtr);
}

} // namespace

namespace fairring {

ProcessGroupFairring::WorkFairring::WorkFairring(
    c10d::OpType opType,
    c10::intrusive_ptr<c10::ivalue::Future> future)
    : c10d::ProcessGroup::Work(/*rank=*/-1, opType),
      future_(std::move(future)) {}

ProcessGroupFairring::WorkFairring::~WorkFairring() {}

bool ProcessGroupFairring::WorkFairring::isCompleted() {
  return future_->completed();
}

bool ProcessGroupFairring::WorkFairring::isSuccess() const {
  return future_->hasValue();
}

std::exception_ptr ProcessGroupFairring::WorkFairring::exception() const {
  return future_->exception_ptr();
}

std::vector<at::Tensor> ProcessGroupFairring::WorkFairring::result() {
  future_->wait();
  return future_->value().toTensorVector();
}

void ProcessGroupFairring::WorkFairring::synchronize() {
  future_->wait();
}

bool ProcessGroupFairring::WorkFairring::wait(
    std::chrono::milliseconds /* timeout */) {
  future_->wait();
  return true;
}

void ProcessGroupFairring::WorkFairring::abort() {}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupFairring::WorkFairring::
    getFuture() {
  return future_;
}

ProcessGroupFairring::OptionsFairring::OptionsFairring(
    int64_t maxMemoryAllocatedInBytes,
    int64_t maxPaddingAllocatedInBytes,
    int64_t minParallelism,
    bool isHighPriorityStream,
    std::chrono::milliseconds timeout)
    : c10d::ProcessGroup::Options(kPgName, timeout),
      maxMemoryAllocatedInBytes_(maxMemoryAllocatedInBytes),
      maxPaddingAllocatedInBytes_(maxPaddingAllocatedInBytes),
      minParallelism_(minParallelism),
      isHighPriorityStream_(isHighPriorityStream) {}

ProcessGroupFairring::OptionsFairring::~OptionsFairring() {}

c10::intrusive_ptr<ProcessGroupFairring::OptionsFairring> ProcessGroupFairring::
    OptionsFairring::create(
        int64_t maxMemoryAllocatedInBytes,
        int64_t maxPaddingAllocatedInBytes,
        int64_t minParallelism,
        bool isHighPriorityStream,
        std::chrono::milliseconds timeout) {
  return c10::make_intrusive<ProcessGroupFairring::OptionsFairring>(
      maxMemoryAllocatedInBytes,
      maxPaddingAllocatedInBytes,
      minParallelism,
      isHighPriorityStream,
      timeout);
}

ProcessGroupFairring::ProcessGroupFairring(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<OptionsFairring> options)
    : ProcessGroup(rank, size),
      store_(store),
      isHighPriorityStream_(options->isHighPriorityStream_),
      maxMemoryAllocatedInBytes_(options->maxMemoryAllocatedInBytes_),
      maxPaddingAllocatedInBytes_(options->maxPaddingAllocatedInBytes_),
      minParallelism_(options->minParallelism_) {}

ProcessGroupFairring::~ProcessGroupFairring() {}

const std::string ProcessGroupFairring::getBackendName() const {
  return kPgName;
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::broadcast(
    std::vector<at::Tensor>& data,
    const c10d::BroadcastOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::BROADCAST, getNcclPG().broadcast(data, opts)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::allreduce(
    std::vector<at::Tensor>& data,
    const c10d::AllreduceOptions& opts) {
  for (at::Tensor& t : data) {
    MY_CHECK(t.layout() == at::kStrided);
    MY_CHECK(t.is_cuda());
    MY_CHECK(t.is_non_overlapping_and_dense());
    t = viewAsFlat(t);
  }
  if (machine_ == nullptr) {
    std::set<c10::DeviceIndex> deviceSet;
    for (const at::Tensor& t : data) {
      if (t.is_cuda()) {
        deviceSet.insert(t.device().index());
      }
    }
    std::vector<c10::Device> devices;
    for (const c10::DeviceIndex& idx : deviceSet) {
      devices.emplace_back(c10::kCUDA, idx);
    }
    initFairring(std::move(devices));
  }
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLREDUCE, machine_->allReduce(opts.reduceOp, data));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceCoalescedOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLREDUCE_COALESCED,
      getNcclPG().allreduce_coalesced(tensors, opts)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::REDUCE, getNcclPG().reduce(tensors, opts)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& /* opts */) {
  MY_CHECK(inputTensors.size() == outputTensors.size());
  int64_t numDevicesPerRank = inputTensors.size();
  std::vector<fairring::MachineFairring::TensorPair> data;
  for (const auto deviceOffset : c10::irange(numDevicesPerRank)) {
    MY_CHECK(
        static_cast<int64_t>(outputTensors[deviceOffset].size()) ==
        size_ * numDevicesPerRank);
    MY_CHECK(inputTensors[deviceOffset].layout() == at::kStrided);
    MY_CHECK(inputTensors[deviceOffset].is_cuda());
    MY_CHECK(inputTensors[deviceOffset].is_non_overlapping_and_dense());
    std::vector<at::Tensor> flattened;
    for (const at::Tensor& t : outputTensors[deviceOffset]) {
      MY_CHECK(t.layout() == at::kStrided);
      MY_CHECK(t.is_cuda());
      MY_CHECK(t.is_non_overlapping_and_dense());
      MY_CHECK(t.device() == inputTensors[deviceOffset].device());
      MY_CHECK(t.scalar_type() == inputTensors[deviceOffset].scalar_type());
      MY_CHECK(t.numel() == inputTensors[deviceOffset].numel());
      flattened.push_back(viewAsFlat(t));
    }
    data.push_back(fairring::MachineFairring::TensorPair{
        .input = viewAsFlat(inputTensors[deviceOffset]),
        .output = unUnbind(std::move(flattened))});
  }
  if (machine_ == nullptr) {
    std::set<c10::DeviceIndex> deviceSet;
    for (const fairring::MachineFairring::TensorPair& pair : data) {
      if (pair.input.is_cuda()) {
        deviceSet.insert(pair.input.device().index());
      }
    }
    std::vector<c10::Device> devices;
    for (const c10::DeviceIndex& idx : deviceSet) {
      devices.emplace_back(c10::kCUDA, idx);
    }
    initFairring(std::move(devices));
  }
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLGATHER, machine_->allGather(std::move(data)));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    _allgather_base(
        at::Tensor& outputBuffer,
        at::Tensor& inputBuffer,
        const c10d::AllgatherOptions& /* opts */) {
  MY_CHECK(inputBuffer.layout() == at::kStrided);
  MY_CHECK(inputBuffer.is_cuda());
  MY_CHECK(inputBuffer.is_non_overlapping_and_dense());
  MY_CHECK(outputBuffer.layout() == at::kStrided);
  MY_CHECK(outputBuffer.is_cuda());
  MY_CHECK(outputBuffer.is_non_overlapping_and_dense());
  MY_CHECK(inputBuffer.device() == outputBuffer.device());
  MY_CHECK(inputBuffer.scalar_type() == outputBuffer.scalar_type());
  MY_CHECK(inputBuffer.numel() * size_ == outputBuffer.numel());
  std::vector<fairring::MachineFairring::TensorPair> data = {
      fairring::MachineFairring::TensorPair{
          .input = viewAsFlat(inputBuffer),
          .output = viewAsFlat(outputBuffer)}};
  if (machine_ == nullptr) {
    std::set<c10::DeviceIndex> deviceSet;
    for (const fairring::MachineFairring::TensorPair& pair : data) {
      if (pair.input.is_cuda()) {
        deviceSet.insert(pair.input.device().index());
      }
    }
    std::vector<c10::Device> devices;
    for (const c10::DeviceIndex& idx : deviceSet) {
      devices.emplace_back(c10::kCUDA, idx);
    }
    initFairring(std::move(devices));
  }
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::_ALLGATHER_BASE, machine_->allGather(std::move(data)));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    allgather_coalesced(
        std::vector<std::vector<at::Tensor>>& outputTensorLists,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLGATHER_COALESCED,
      getNcclPG()
          .allgather_coalesced(outputTensorLists, inputTensors, opts)
          ->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::GATHER,
      getNcclPG().gather(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::SCATTER,
      getNcclPG().scatter(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ReduceScatterOptions& opts) {
  MY_CHECK(inputTensors.size() == outputTensors.size());
  int64_t numDevicesPerRank = outputTensors.size();
  std::vector<fairring::MachineFairring::TensorPair> data;
  for (const auto deviceOffset : c10::irange(numDevicesPerRank)) {
    MY_CHECK(
        static_cast<int64_t>(inputTensors[deviceOffset].size()) ==
        size_ * numDevicesPerRank);
    MY_CHECK(outputTensors[deviceOffset].layout() == at::kStrided);
    MY_CHECK(outputTensors[deviceOffset].is_cuda());
    MY_CHECK(outputTensors[deviceOffset].is_non_overlapping_and_dense());
    std::vector<at::Tensor> flattened;
    for (const at::Tensor& t : inputTensors[deviceOffset]) {
      MY_CHECK(t.layout() == at::kStrided);
      MY_CHECK(t.is_cuda());
      MY_CHECK(t.is_non_overlapping_and_dense());
      MY_CHECK(t.device() == outputTensors[deviceOffset].device());
      MY_CHECK(t.scalar_type() == outputTensors[deviceOffset].scalar_type());
      MY_CHECK(t.numel() == outputTensors[deviceOffset].numel());
      flattened.push_back(viewAsFlat(t));
    }
    data.push_back(fairring::MachineFairring::TensorPair{
        .input = unUnbind(std::move(flattened)),
        .output = viewAsFlat(outputTensors[deviceOffset])});
  }
  if (machine_ == nullptr) {
    std::set<c10::DeviceIndex> deviceSet;
    for (const fairring::MachineFairring::TensorPair& pair : data) {
      if (pair.output.is_cuda()) {
        deviceSet.insert(pair.output.device().index());
      }
    }
    std::vector<c10::Device> devices;
    for (const c10::DeviceIndex& idx : deviceSet) {
      devices.emplace_back(c10::kCUDA, idx);
    }
    initFairring(std::move(devices));
  }
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::REDUCE_SCATTER,
      machine_->reduceScatter(opts.reduceOp, std::move(data)));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    _reduce_scatter_base(
        at::Tensor& outputBuffer,
        at::Tensor& inputBuffer,
        const c10d::ReduceScatterOptions& opts) {
  MY_CHECK(inputBuffer.layout() == at::kStrided);
  MY_CHECK(inputBuffer.is_cuda());
  MY_CHECK(inputBuffer.is_non_overlapping_and_dense());
  MY_CHECK(outputBuffer.layout() == at::kStrided);
  MY_CHECK(outputBuffer.is_cuda());
  MY_CHECK(outputBuffer.is_non_overlapping_and_dense());
  MY_CHECK(inputBuffer.device() == outputBuffer.device());
  MY_CHECK(inputBuffer.scalar_type() == outputBuffer.scalar_type());
  MY_CHECK(inputBuffer.numel() == outputBuffer.numel() * size_);
  std::vector<fairring::MachineFairring::TensorPair> data = {
      fairring::MachineFairring::TensorPair{
          .input = viewAsFlat(inputBuffer),
          .output = viewAsFlat(outputBuffer)}};
  if (machine_ == nullptr) {
    std::set<c10::DeviceIndex> deviceSet;
    for (const fairring::MachineFairring::TensorPair& pair : data) {
      if (pair.output.is_cuda()) {
        deviceSet.insert(pair.output.device().index());
      }
    }
    std::vector<c10::Device> devices;
    for (const c10::DeviceIndex& idx : deviceSet) {
      devices.emplace_back(c10::kCUDA, idx);
    }
    initFairring(std::move(devices));
  }
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::_REDUCE_SCATTER_BASE,
      machine_->reduceScatter(opts.reduceOp, std::move(data)));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const c10d::AllToAllOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLTOALL_BASE,
      ncclPG_
          ->alltoall_base(
              outputTensor,
              inputTensor,
              outputSplitSizes,
              inputSplitSizes,
              opts)
          ->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::ALLTOALL,
      getNcclPG().alltoall(outputTensors, inputTensors, opts)->getFuture());
}

void ProcessGroupFairring::monitoredBarrier(
    const c10d::BarrierOptions& opts,
    bool waitAllRanks) {
  getNcclPG().monitoredBarrier(opts, waitAllRanks);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::SEND, getNcclPG().send(tensors, dstRank, tag)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::RECV, getNcclPG().recv(tensors, srcRank, tag)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::
    recvAnysource(std::vector<at::Tensor>& tensors, int tag) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::RECVANYSOURCE,
      getNcclPG().recvAnysource(tensors, tag)->getFuture());
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupFairring::barrier(
    const c10d::BarrierOptions& opts) {
  return c10::make_intrusive<WorkFairring>(
      c10d::OpType::BARRIER, getNcclPG().barrier(opts)->getFuture());
}

c10d::ProcessGroupNCCL& ProcessGroupFairring::getNcclPG() {
  if (!ncclPG_) {
    ncclPG_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        c10::make_intrusive<c10d::PrefixStore>("nccl", store_),
        rank_,
        size_,
        c10d::ProcessGroupNCCL::Options::create(isHighPriorityStream_));
  }
  return *ncclPG_;
}

void ProcessGroupFairring::initFairring(std::vector<c10::Device> devices) {
  machine_ = std::make_unique<fairring::MachineFairring>(
      c10::make_intrusive<c10d::PrefixStore>("fairring", store_),
      rank_,
      size_,
      std::move(devices),
      maxMemoryAllocatedInBytes_,
      maxPaddingAllocatedInBytes_,
      minParallelism_);
}

} // namespace fairring
