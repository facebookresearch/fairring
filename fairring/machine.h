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
  std::vector<std::unique_ptr<DeviceFairring>> nodes_;
  std::vector<c10::Device> devices_;
  std::vector<CudaStream> streams_;
  std::unordered_map<c10::Device, int64_t> deviceToOffset_;

  c10::intrusive_ptr<c10::ivalue::Future> mergeMultiDeviceFutures(
      c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures);
};

} // namespace fairring
