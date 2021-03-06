/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <torch/torch.h>

#include <fairring/utils.h>

namespace fairring {

class DeviceFairring;

struct DeploymentInfo {
  int64_t numDevices;
  int64_t numMachines;
  int64_t idxOfMyFirstDevice;
  int64_t idxOfMyMachine;
  int64_t devicesPerMachine;
  int64_t deviceGlobalRankIsFavorable;
};

DeploymentInfo detectDeploymentInfo(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    int64_t numDevices);

class MachineFairring {
 public:
  MachineFairring(
      c10::intrusive_ptr<c10d::Store> store,
      int rank,
      int size,
      std::vector<c10::Device> devices,
      int64_t maxMemoryAllocatedInBytes,
      int64_t maxPaddingAllocatedInBytes,
      int64_t minParallelism);

  ~MachineFairring();

  c10::intrusive_ptr<c10::ivalue::Future> allReduce(
      c10d::ReduceOp opType,
      std::vector<at::Tensor> tensors);

  struct TensorPair {
    at::Tensor input;
    at::Tensor output;
  };

  c10::intrusive_ptr<c10::ivalue::Future> reduceScatter(
      c10d::ReduceOp opType,
      std::vector<TensorPair> tensors);

  c10::intrusive_ptr<c10::ivalue::Future> allGather(
      std::vector<TensorPair> tensors);

 private:
  c10::intrusive_ptr<c10d::Store> store_;
  DeploymentInfo deploymentInfo_;
  std::vector<std::unique_ptr<DeviceFairring>> nodes_;
  std::vector<c10::Device> devices_;
  std::vector<CudaStream> streams_;
  std::unordered_map<c10::Device, int64_t> deviceToOffset_;

  bool establishedReduceScatterComm_ = false;
  bool establishedCollectComm_ = false;
  bool establishedDiffuseComm_ = false;
  bool establishedAllGatherComm_ = false;

  std::vector<NcclComm> establishReduceScatterComms();
  std::vector<NcclComm> establishCollectComms();
  std::vector<NcclComm> establishDiffuseComms();
  std::vector<NcclComm> establishAllGatherComms();

  void ensureReduceScatterCommsEstablished();
  void ensureCollectCommsEstablished();
  void ensureDiffuseCommsEstablished();
  void ensureAllGatherCommsEstablished();

  c10::intrusive_ptr<c10::ivalue::Future> mergeMultiDeviceFutures(
      c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures);
};

} // namespace fairring
