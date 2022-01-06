/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/Store.hpp>
#include <torch/torch.h>

namespace fairring {

constexpr const char* kPgName = "fairring";

class AllReduceFairring;

class ProcessGroupFairring : public c10d::ProcessGroup {
 public:
  class WorkFairring : public c10d::ProcessGroup::Work {
   public:
    explicit WorkFairring(
        c10d::OpType opType,
        c10::intrusive_ptr<c10::ivalue::Future> future);

    ~WorkFairring() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    std::exception_ptr exception() const override;

    std::vector<at::Tensor> result() override;

    void synchronize() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   private:
    const c10::intrusive_ptr<c10::ivalue::Future> future_;
  };

  class OptionsFairring : public c10d::ProcessGroup::Options {
   public:
    explicit OptionsFairring(
        size_t maxMemoryAllocatedInBytes,
        size_t maxPaddingAllocatedInBytes,
        size_t minParallelism,
        bool isHighPriorityStream = false,
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout);

    ~OptionsFairring() override;

    static c10::intrusive_ptr<OptionsFairring> create(
        size_t maxMemoryAllocatedInBytes,
        size_t maxPaddingAllocatedInBytes,
        size_t minParallelism,
        bool isHighPriorityStream = false,
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout);

   private:
    const size_t maxMemoryAllocatedInBytes_;
    const size_t maxPaddingAllocatedInBytes_;
    const size_t minParallelism_;
    const bool isHighPriorityStream_;

    friend ProcessGroupFairring;
  };

  ProcessGroupFairring(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<OptionsFairring> options);

  ~ProcessGroupFairring() override;

  const std::string getBackendName() const override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& data,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> _reduce_scatter_base(
      at::Tensor&,
      at::Tensor&,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  void monitoredBarrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions(),
      bool waitAllRanks = false) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

 private:
  c10::intrusive_ptr<c10d::ProcessGroupNCCL> ncclPG_;

  size_t maxMemoryAllocatedInBytes_;
  size_t maxPaddingAllocatedInBytes_;
  size_t minParallelism_;
  c10::intrusive_ptr<c10d::Store> store_;
  std::unique_ptr<AllReduceFairring> allReduce_;
};

} // namespace fairring
