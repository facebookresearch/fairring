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
      size_t deviceIdxOnProcess,
      size_t machineIdx,
      size_t deviceIdxOnMachine,
      size_t numMachines,
      size_t numDevicesPerMachine,
      c10::intrusive_ptr<c10d::Store> store,
      NcclComm reduceScatterComm,
      NcclComm collectComm,
      NcclComm diffuseComm,
      NcclComm allGatherComm,
      size_t maxMemoryAllocatedInBytes,
      size_t sliceSizeInBytes);

  ~DeviceFairring();

  c10::intrusive_ptr<c10::ivalue::Future> allReduce(
      c10d::ReduceOp opType,
      at::Tensor tensor);

 private:
  // Common

  size_t myDeviceIdxOnProcess_;
  size_t myMachineIdx_;
  size_t myDeviceIdxOnMachine_;
  size_t numMachines_;
  size_t numDevicesPerMachine_;
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
  at::Tensor diffusedBuffer_;
  at::Tensor collectedBuffer_;
  at::Tensor reducedBuffer_;

  // To sync the "diffused" buffer across slots.
  std::vector<at::cuda::CUDAEvent> allgatherToReduceScatterEvents_;
  // To sync the "collected" buffer across slots.
  std::vector<at::cuda::CUDAEvent> addToCollectEvents_;
  // To sync the "reduced" buffer across slots.
  std::vector<at::cuda::CUDAEvent> diffuseToAddEvents_;

  CommandQueue cmdQueue_;
  std::thread cmdThread_;
  size_t nextSlot_{0};

  void processOneSlice(
      size_t seqNum,
      at::Tensor slice,
      c10::optional<at::cuda::CUDAEvent> initialEvent);
};

} // namespace fairring
