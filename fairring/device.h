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
#include <tensorpipe/tensorpipe.h>
#include <torch/torch.h>

#include <fairring/tpcoro.h>
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
      NcclComm allGatherComm,
      size_t maxMemoryAllocatedInBytes,
      size_t sliceSizeInBytes);

  // Initializing the nodes is tricky: we can only connect to another node once
  // it has started listening (or else it will fail with a connection error) and
  // we can only accept a pipe (and block until that completes) once the other
  // endpoint has started connecting. This gets trickier with the fact that a
  // single process might host multiple nodes. Hence we provide methods to do
  // each of these steps, and users should call the first method on all nodes
  // before calling the second on any of them, and so on.
  void startListening();
  void connect();
  void acceptPipes();

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
  NcclComm allGatherComm_;
  Layout layout_;

  std::vector<int64_t> remoteDeviceIndices_;

  std::shared_ptr<tensorpipe::Context> tpCtx_;
  std::shared_ptr<tensorpipe::Listener> tpListener_;
  std::vector<std::shared_ptr<tensorpipe::Pipe>> leafTpPipes_;
  std::vector<std::shared_ptr<tensorpipe::Pipe>> rootTpPipes_;

  std::vector<tpcoro::Sequencer> slotSequencers_;
  tpcoro::Sequencer reduceScatterSequencer_;
  std::vector<tpcoro::Sequencer> leafTpWriteSequencers_;
  std::vector<tpcoro::Sequencer> rootTpReadDescriptorSequencers_;
  std::vector<tpcoro::Sequencer> rootTpReadSequencers_;
  std::vector<tpcoro::Sequencer> rootTpWriteSequencers_;
  std::vector<tpcoro::Sequencer> leafTpReadDescriptorSequencers_;
  std::vector<tpcoro::Sequencer> leafTpReadSequencers_;
  tpcoro::Sequencer allGatherSequencer_;

  CudaStream reduceScatterStream_;
  std::vector<CudaStream> leafCollectStreams_;
  std::vector<CudaStream> rootCollectStreams_;
  CudaStream addStream_;
  std::vector<CudaStream> rootDiffuseStreams_;
  std::vector<CudaStream> leafDiffuseStreams_;
  CudaStream allGatherStream_;

  at::Tensor paddingBuffer_;
  at::Tensor diffusedBuffer_;
  at::Tensor collectedBuffer_;
  at::Tensor reducedBuffer_;

  // To sync the "diffused" buffer across slots.
  std::vector<at::cuda::CUDAEvent> allgatherToReduceScatterEvents_;
  // To sync the "collected" buffer across slots.
  std::vector<std::vector<at::cuda::CUDAEvent>> addToCollectEvents_;
  // To sync the "reduced" buffer across slots.
  std::vector<std::vector<at::cuda::CUDAEvent>> diffuseToAddEvents_;

  tpcoro::CoroTracker tracker_;

  CommandQueue cmdQueue_;
  std::thread cmdThread_;
  size_t nextSlot_{0};

  tpcoro::Task processOneSlice(
      size_t seqNum,
      at::Tensor slice,
      c10::optional<at::cuda::CUDAEvent> initialEvent);
};

} // namespace fairring
