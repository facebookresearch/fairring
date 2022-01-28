/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <ATen/cuda/CUDAEvent.h>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <torch/torch.h>

#include <fairring/utils.h>

namespace fairring {

class DeviceFairring {
 public:
  DeviceFairring(
      int64_t deviceIdxOnProcess,
      int64_t machineIdx,
      int64_t deviceIdxOnMachine,
      int64_t numMachines,
      int64_t numDevicesPerMachine,
      int64_t deviceGlobalRankIsFavorable,
      c10::intrusive_ptr<c10d::Store> store,
      int64_t maxMemoryAllocatedInBytes,
      int64_t maxPaddingAllocatedInBytes,
      int64_t minParallelism);

  void setReduceScatterComm(NcclComm reduceScatterComm) {
    reduceScatterComm_ = std::move(reduceScatterComm);
  }

  void setCollectComm(NcclComm collectComm) {
    collectComm_ = std::move(collectComm);
  }

  void setDiffuseComm(NcclComm diffuseComm) {
    diffuseComm_ = std::move(diffuseComm);
  }

  void setAllGatherComm(NcclComm allGatherComm) {
    allGatherComm_ = std::move(allGatherComm);
  }

  ~DeviceFairring();

  c10::intrusive_ptr<c10::ivalue::Future> allReduce(
      c10d::ReduceOp opType,
      at::Tensor tensor);

  c10::intrusive_ptr<c10::ivalue::Future> reduceScatter(
      c10d::ReduceOp opType,
      at::Tensor input,
      at::Tensor output);

  c10::intrusive_ptr<c10::ivalue::Future> allGather(
      at::Tensor input,
      at::Tensor output);

 private:
  // Common

  int64_t myDeviceIdxOnProcess_;
  int64_t myMachineIdx_;
  int64_t myDeviceIdxOnMachine_;
  int64_t numMachines_;
  int64_t numDevicesPerMachine_;
  int64_t deviceGlobalRankIsFavorable_;
  c10::intrusive_ptr<c10d::Store> store_;
  NcclComm reduceScatterComm_;
  NcclComm collectComm_;
  NcclComm diffuseComm_;
  NcclComm allGatherComm_;
  Layout layout_;

  CudaStream reduceScatterStream_;
  CudaStream collectStream_;
  CudaStream addStream_;
  CudaStream diffuseStream_;
  CudaStream allGatherStream_;

  at::Tensor paddingBuffer_;
  std::vector<at::cuda::CUDAEvent> paddingEvents_;
  int64_t nextPaddingSlot_{0};

  at::Tensor stagingBuffer_;
  at::Tensor paddingStagingBuffer_;
  std::vector<at::cuda::CUDAEvent> stagingEvents_;
  int64_t nextStagingSlot_{0};

  CommandQueue cmdQueue_;
  std::thread cmdThread_;
  int64_t nextSlot_{0};

  void allReduceOneSlice(
      c10d::ReduceOp opType,
      at::Tensor slice,
      c10::optional<at::cuda::CUDAEvent> initialEvent);

  void reduceScatterOneSlice(
      c10d::ReduceOp opType,
      at::Tensor input,
      at::Tensor output,
      at::cuda::CUDAEvent initialEvent);

  void allGatherOneSlice(
      at::Tensor input,
      at::Tensor output,
      at::cuda::CUDAEvent initialEvent);
};

} // namespace fairring
