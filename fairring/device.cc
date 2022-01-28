/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fairring/device.h>

#include <future>

#include <ATen/Functions.h>

namespace fairring {

namespace {

std::vector<CudaStream> makeManyCudaStreams(int64_t amount, int deviceIdx) {
  std::vector<CudaStream> result;
  result.reserve(amount);
  for (const auto _ : c10::irange(amount)) {
    result.push_back(CudaStream(deviceIdx));
  }
  return result;
}

std::vector<at::cuda::CUDAEvent> makeManyCudaEvents(int n) {
  return std::vector<at::cuda::CUDAEvent>(n);
}

template <typename... T>
auto makeManyCudaEvents(int n, T... args) {
  std::vector<decltype(makeManyCudaEvents(args...))> result;
  result.reserve(n);
  for (const auto idx : c10::irange(n)) {
    result.push_back(makeManyCudaEvents(args...));
  }
  return result;
}

void doReduceScatter(
    at::Tensor t,
    int64_t myRank,
    c10d::ReduceOp opType,
    NcclComm& comm,
    CudaStream& stream) {
  MY_CHECK(t.layout() == at::kStrided);
  MY_CHECK(t.is_non_overlapping_and_dense());
  MY_CHECK(t.dim() >= 1);
  MY_CHECK((0 <= myRank) && (myRank < t.size(0)));
  if (t.numel() == 0) {
    return;
  }
  if (t.is_contiguous()) {
    NCCL_CHECK(ncclReduceScatter(
        t.data_ptr(),
        t[myRank].data_ptr(),
        t[myRank].numel(),
        torchToNcclDtype(t.scalar_type()),
        torchToNcclRedOp(opType),
        comm.get(),
        stream));
  } else {
    MY_CHECK(t.dim() >= 2);
    t = t.transpose(0, 1);
    MY_CHECK(t.is_contiguous());
    for (const auto idx : c10::irange(t.size(0))) {
      NCCL_CHECK(ncclReduceScatter(
          t[idx].data_ptr(),
          t[idx][myRank].data_ptr(),
          t[idx][myRank].numel(),
          torchToNcclDtype(t.scalar_type()),
          torchToNcclRedOp(opType),
          comm.get(),
          stream));
    }
  }
}

void doCollect(
    at::Tensor sendT,
    at::Tensor recvT,
    int64_t myRank,
    NcclComm& comm,
    CudaStream& stream) {
  MY_CHECK(sendT.layout() == at::kStrided);
  MY_CHECK(recvT.layout() == at::kStrided);
  MY_CHECK(sendT.sizes() == recvT.sizes());
  MY_CHECK(sendT.strides() == recvT.strides());
  MY_CHECK(sendT.dim() >= 1);
  MY_CHECK((0 <= myRank) && (myRank < sendT.size(0)));
  MY_CHECK(sendT[0].is_non_overlapping_and_dense());
  if (sendT.numel() == 0) {
    return;
  }
  for (const auto otherRank : c10::irange(sendT.size(0))) {
    NCCL_CHECK(ncclSend(
        sendT[otherRank].data_ptr(),
        sendT[otherRank].numel(),
        torchToNcclDtype(sendT.scalar_type()),
        otherRank,
        comm.get(),
        stream));
  }
  for (const auto otherRank : c10::irange(recvT.size(0))) {
    NCCL_CHECK(ncclRecv(
        recvT[otherRank].data_ptr(),
        recvT[otherRank].numel(),
        torchToNcclDtype(recvT.scalar_type()),
        otherRank,
        comm.get(),
        stream));
  }
}

void doReduction(
    at::Tensor operand,
    at::Tensor result,
    c10d::ReduceOp opType,
    int64_t myRank,
    CudaStream& stream) {
  MY_CHECK(operand.layout() == at::kStrided);
  MY_CHECK(result.layout() == at::kStrided);
  MY_CHECK(operand.dim() >= 1);
  MY_CHECK((0 <= myRank) && (myRank < operand.size(0)));
  MY_CHECK(result.sizes() == operand[0].sizes());
  MY_CHECK(result.strides() == operand[0].strides());
  if (operand.numel() == 0) {
    return;
  }
  c10::cuda::CUDAStreamGuard g(stream);
  if (opType == c10d::ReduceOp::SUM) {
    at::sum_out(result, operand, {0});
  } else if (opType == c10d::ReduceOp::MAX) {
    at::amax_out(result, operand, {0});
  } else {
    MY_CHECK(false);
  }
}

void doDiffuse(
    at::Tensor sendT,
    at::Tensor recvT,
    int64_t myRank,
    NcclComm& comm,
    CudaStream& stream) {
  MY_CHECK(sendT.layout() == at::kStrided);
  MY_CHECK(recvT.layout() == at::kStrided);
  MY_CHECK(recvT.dim() >= 1);
  MY_CHECK((0 <= myRank) && (myRank < recvT.size(0)));
  MY_CHECK(sendT.sizes() == recvT[0].sizes());
  MY_CHECK(sendT.strides() == recvT[0].strides());
  MY_CHECK(recvT[0].is_non_overlapping_and_dense());
  if (recvT.numel() == 0) {
    return;
  }
  for (const auto otherRank : c10::irange(recvT.size(0))) {
    if (otherRank == myRank && recvT[myRank].data_ptr() == sendT.data_ptr()) {
      continue;
    }
    NCCL_CHECK(ncclSend(
        sendT.data_ptr(),
        sendT.numel(),
        torchToNcclDtype(recvT.scalar_type()),
        otherRank,
        comm.get(),
        stream));
  }
  for (const auto otherRank : c10::irange(recvT.size(0))) {
    if (otherRank == myRank && recvT[myRank].data_ptr() == sendT.data_ptr()) {
      continue;
    }
    NCCL_CHECK(ncclRecv(
        recvT[otherRank].data_ptr(),
        recvT[otherRank].numel(),
        torchToNcclDtype(recvT.scalar_type()),
        otherRank,
        comm.get(),
        stream));
  }
}

void doAllGather(
    at::Tensor t,
    int64_t myRank,
    NcclComm& comm,
    CudaStream& stream) {
  MY_CHECK(t.layout() == at::kStrided);
  MY_CHECK(t.is_non_overlapping_and_dense());
  MY_CHECK(t.dim() >= 1);
  MY_CHECK((0 <= myRank) && (myRank < t.size(0)));
  if (t.numel() == 0) {
    return;
  }
  if (t.is_contiguous()) {
    NCCL_CHECK(ncclAllGather(
        t[myRank].data_ptr(),
        t.data_ptr(),
        t[myRank].numel(),
        torchToNcclDtype(t.scalar_type()),
        comm.get(),
        stream));
  } else {
    MY_CHECK(t.dim() >= 2);
    t = t.transpose(0, 1);
    MY_CHECK(t.is_contiguous());
    for (const auto idx : c10::irange(t.size(0))) {
      NCCL_CHECK(ncclAllGather(
          t[idx][myRank].data_ptr(),
          t[idx].data_ptr(),
          t[idx][myRank].numel(),
          torchToNcclDtype(t.scalar_type()),
          comm.get(),
          stream));
    }
  }
}

} // namespace

DeviceFairring::DeviceFairring(
    int64_t deviceIdxOnProcess,
    int64_t machineIdx,
    int64_t deviceIdxOnMachine,
    int64_t numMachines,
    int64_t numDevicesPerMachine,
    int64_t deviceGlobalRankIsFavorable,
    c10::intrusive_ptr<c10d::Store> store,
    int64_t maxMemoryAllocatedInBytes,
    int64_t maxPaddingAllocatedInBytes,
    int64_t minParallelism)
    : // Arguments
      myDeviceIdxOnProcess_(deviceIdxOnProcess),
      myMachineIdx_(machineIdx),
      myDeviceIdxOnMachine_(deviceIdxOnMachine),
      numMachines_(numMachines),
      numDevicesPerMachine_(numDevicesPerMachine),
      deviceGlobalRankIsFavorable_(deviceGlobalRankIsFavorable),
      store_(std::move(store)),
      layout_(computeLayout(
          maxMemoryAllocatedInBytes,
          maxPaddingAllocatedInBytes,
          minParallelism,
          numMachines_,
          numDevicesPerMachine_)),
      // Streams
      reduceScatterStream_(CudaStream(myDeviceIdxOnProcess_)),
      collectStream_(CudaStream(myDeviceIdxOnProcess_)),
      addStream_(CudaStream(myDeviceIdxOnProcess_)),
      diffuseStream_(CudaStream(myDeviceIdxOnProcess_)),
      allGatherStream_(CudaStream(myDeviceIdxOnProcess_)),
      // Padding
      paddingBuffer_(at::empty(
          {layout_.numPaddingSlots,
           numDevicesPerMachine_,
           numMachines_,
           kAlignment},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      paddingEvents_(makeManyCudaEvents(layout_.numPaddingSlots)),
      // Staging
      stagingBuffer_(at::empty(
          {layout_.numStagingSlots, numMachines_, layout_.slotSizeInBytes},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      paddingStagingBuffer_(at::empty(
          {layout_.numStagingSlots, numMachines_, kAlignment},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      stagingEvents_(makeManyCudaEvents(layout_.numStagingSlots)) {
  cmdThread_ = std::thread([this]() {
    while (true) {
      std::function<void()> fn = cmdQueue_.dequeue();
      if (fn == nullptr) {
        return;
      }
      fn();
    }
  });
}

DeviceFairring::~DeviceFairring() {
  cmdQueue_.enqueue(nullptr);
  cmdThread_.join();
}

c10::intrusive_ptr<c10::ivalue::Future> DeviceFairring::allReduce(
    c10d::ReduceOp opType,
    at::Tensor tensor) {
  at::cuda::CUDAEvent initialEvent;
  initialEvent.record(c10::cuda::getCurrentCUDAStream(myDeviceIdxOnProcess_));

  c10::intrusive_ptr<c10::ivalue::Future> future =
      c10::make_intrusive<c10::ivalue::Future>(
          c10::ListType::ofTensors(),
          std::vector<c10::Device>(
              {c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)}));

  cmdQueue_.enqueue([this,
                     opType,
                     tensor = std::move(tensor),
                     initialEvent = std::make_shared<at::cuda::CUDAEvent>(
                         std::move(initialEvent)),
                     future]() mutable {
    int64_t numElements = tensor.numel();
    MY_CHECK(kAlignment % tensor.element_size() == 0);
    int64_t maxSliceSizeInElems =
        layout_.sliceSizeInBytes / tensor.element_size();
    int64_t numSlices = ceilOfRatio(numElements, maxSliceSizeInElems);
    for (const auto sliceIdx : c10::irange(numSlices)) {
      int64_t seqNum = nextSlot_++;
      int64_t offsetInElems = sliceIdx * maxSliceSizeInElems;
      int64_t sliceSizeInElems =
          std::min(maxSliceSizeInElems, numElements - offsetInElems);
      at::Tensor slice = tensor.slice(
          /*dim=*/0, offsetInElems, offsetInElems + sliceSizeInElems);
      auto myFuture = sliceIdx == numSlices - 1
          ? std::move(future)
          : c10::intrusive_ptr<c10::ivalue::Future>();
      try {
        allReduceOneSlice(
            opType,
            std::move(slice),
            sliceIdx == 0
                ? c10::optional<at::cuda::CUDAEvent>(std::move(*initialEvent))
                : c10::nullopt);
        if (myFuture) {
          c10::cuda::CUDAStreamGuard g(allGatherStream_);
          myFuture->markCompleted(std::vector<at::Tensor>{std::move(tensor)});
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Function for chunk #" << seqNum
                   << " threw exception: " << e.what();
        if (myFuture) {
          c10::cuda::CUDAStreamGuard g(allGatherStream_);
          myFuture->setError(std::current_exception());
        }
      }
    }
  });

  return future;
}

c10::intrusive_ptr<c10::ivalue::Future> DeviceFairring::reduceScatter(
    c10d::ReduceOp opType,
    at::Tensor input,
    at::Tensor output) {
  // Because reduce-scatter cannot be sliced
  MY_CHECK(numDevicesPerMachine_ >= 2);

  at::cuda::CUDAEvent initialEvent;
  initialEvent.record(c10::cuda::getCurrentCUDAStream(myDeviceIdxOnProcess_));

  c10::intrusive_ptr<c10::ivalue::Future> future =
      c10::make_intrusive<c10::ivalue::Future>(
          c10::ListType::ofTensors(),
          std::vector<c10::Device>(
              {c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)}));

  cmdQueue_.enqueue([this,
                     opType,
                     input = std::move(input),
                     output = std::move(output),
                     initialEvent = std::make_shared<at::cuda::CUDAEvent>(
                         std::move(initialEvent)),
                     future]() mutable {
    MY_CHECK(kAlignment % input.element_size() == 0);
    int64_t seqNum = nextSlot_++;
    try {
      reduceScatterOneSlice(opType, input, output, std::move(*initialEvent));
      c10::cuda::CUDAStreamGuard g(addStream_);
      future->markCompleted(std::vector<at::Tensor>{output});
    } catch (const std::exception& e) {
      LOG(ERROR) << "Function for chunk #" << seqNum
                 << " threw exception: " << e.what();
      c10::cuda::CUDAStreamGuard g(addStream_);
      future->setError(std::current_exception());
    }
  });

  return future;
}

c10::intrusive_ptr<c10::ivalue::Future> DeviceFairring::allGather(
    at::Tensor input,
    at::Tensor output) {
  // Because all-gather cannot be sliced
  MY_CHECK(numDevicesPerMachine_ >= 2);

  at::cuda::CUDAEvent initialEvent;
  initialEvent.record(c10::cuda::getCurrentCUDAStream(myDeviceIdxOnProcess_));

  c10::intrusive_ptr<c10::ivalue::Future> future =
      c10::make_intrusive<c10::ivalue::Future>(
          c10::ListType::ofTensors(),
          std::vector<c10::Device>(
              {c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)}));

  cmdQueue_.enqueue([this,
                     input = std::move(input),
                     output = std::move(output),
                     initialEvent = std::make_shared<at::cuda::CUDAEvent>(
                         std::move(initialEvent)),
                     future]() mutable {
    MY_CHECK(kAlignment % input.element_size() == 0);
    int64_t seqNum = nextSlot_++;
    try {
      allGatherOneSlice(input, output, std::move(*initialEvent));
      c10::cuda::CUDAStreamGuard g(allGatherStream_);
      future->markCompleted(std::vector<at::Tensor>{output});
    } catch (const std::exception& e) {
      LOG(ERROR) << "Function for chunk #" << seqNum
                 << " threw exception: " << e.what();
      c10::cuda::CUDAStreamGuard g(allGatherStream_);
      future->setError(std::current_exception());
    }
  });

  return future;
}

void DeviceFairring::allReduceOneSlice(
    c10d::ReduceOp opType,
    at::Tensor slice,
    c10::optional<at::cuda::CUDAEvent> initialEvent) {
  c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);

  c10::ScalarType dtype = slice.scalar_type();
  int64_t elementSizeInBytes = slice.element_size();

  at::cuda::CUDAEvent reduceScatterToCollectEvent;
  at::cuda::CUDAEvent collectToAddEvent;
  at::cuda::CUDAEvent addToDiffuseEvent;
  at::cuda::CUDAEvent diffuseToAllGatherEvent;

  at::Tensor slice3d;
  c10::optional<at::Tensor> padding;
  at::cuda::CUDAEvent* paddingEvent = nullptr;
  if (slice.numel() % (numDevicesPerMachine_ * numMachines_) == 0) {
    slice3d = slice.view({numDevicesPerMachine_, numMachines_, -1});
  } else {
    int64_t sliceSizeInElems = roundDownToNearestMultiple(
        slice.numel(), numDevicesPerMachine_ * numMachines_);
    slice3d = slice.index({torch::indexing::Slice(0, sliceSizeInElems)})
                  .view({numDevicesPerMachine_, numMachines_, -1});
    int64_t paddingSlotIdx = (nextPaddingSlot_++) % layout_.numPaddingSlots;
    padding = paddingBuffer_[paddingSlotIdx]
                  .view(dtype)
                  .flatten()
                  .index({torch::indexing::Slice(
                      0, numDevicesPerMachine_ * numMachines_)})
                  .view({numDevicesPerMachine_, numMachines_});
    paddingEvent = &paddingEvents_[paddingSlotIdx];
  }

  at::Tensor slice3dStaging;
  c10::optional<at::Tensor> paddingStaging;
  at::cuda::CUDAEvent* stagingEvent = nullptr;
  if (numDevicesPerMachine_ == 1) {
    int64_t stagingSlotIdx = (nextStagingSlot_++) % layout_.numStagingSlots;
    slice3dStaging = stagingBuffer_[stagingSlotIdx]
                         .view(dtype)
                         .flatten()
                         .index({torch::indexing::Slice(
                             0, slice3d[myDeviceIdxOnMachine_].numel())})
                         .view({numMachines_, -1});
    if (padding) {
      paddingStaging = paddingStagingBuffer_[stagingSlotIdx]
                           .view(dtype)
                           .flatten()
                           .index({torch::indexing::Slice(
                               0, (*padding)[myDeviceIdxOnMachine_].numel())})
                           .view({numMachines_});
    }
    stagingEvent = &stagingEvents_[stagingSlotIdx];
  } else {
    slice3dStaging =
        slice3d[(myDeviceIdxOnMachine_ + 1) % numDevicesPerMachine_];
    if (padding) {
      paddingStaging =
          (*padding)[(myDeviceIdxOnMachine_ + 1) % numDevicesPerMachine_];
    }
  }

  if (initialEvent.has_value()) {
    initialEvent.value().block(reduceScatterStream_);
  }

  if (padding) {
    (*paddingEvent).block(reduceScatterStream_);
    // No need to zero out the padding: we don't care what value it has/gets.
    CUDA_CHECK(cudaMemcpyAsync(
        (*padding).data_ptr(),
        reinterpret_cast<uint8_t*>(slice.data_ptr()) +
            slice3d.numel() * elementSizeInBytes,
        (slice.numel() - slice3d.numel()) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        reduceScatterStream_));
  }

  if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doReduceScatter(
        slice3d,
        myDeviceIdxOnMachine_,
        opType,
        reduceScatterComm_,
        reduceScatterStream_);
    if (padding) {
      doReduceScatter(
          *padding,
          myDeviceIdxOnMachine_,
          opType,
          reduceScatterComm_,
          reduceScatterStream_);
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  reduceScatterToCollectEvent.record(reduceScatterStream_);

  if (stagingEvent) {
    (*stagingEvent).block(collectStream_);
  }

  reduceScatterToCollectEvent.block(collectStream_);
  if (numMachines_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doCollect(
        slice3d[myDeviceIdxOnMachine_],
        slice3dStaging,
        myMachineIdx_,
        collectComm_,
        collectStream_);
    if (padding) {
      doCollect(
          (*padding)[myDeviceIdxOnMachine_],
          *paddingStaging,
          myMachineIdx_,
          collectComm_,
          collectStream_);
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  collectToAddEvent.record(collectStream_);

  collectToAddEvent.block(addStream_);
  if (numMachines_ > 1) {
    doReduction(
        slice3dStaging,
        slice3d[myDeviceIdxOnMachine_][myMachineIdx_],
        opType,
        myMachineIdx_,
        addStream_);
    if (padding) {
      doReduction(
          *paddingStaging,
          (*padding)[myDeviceIdxOnMachine_][myMachineIdx_],
          opType,
          myMachineIdx_,
          addStream_);
    }
  }
  addToDiffuseEvent.record(addStream_);

  if (stagingEvent) {
    (*stagingEvent).record(addStream_);
  }

  addToDiffuseEvent.block(diffuseStream_);
  if (numMachines_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doDiffuse(
        slice3d[myDeviceIdxOnMachine_][myMachineIdx_],
        slice3d[myDeviceIdxOnMachine_],
        myMachineIdx_,
        diffuseComm_,
        diffuseStream_);
    if (padding) {
      doDiffuse(
          (*padding)[myDeviceIdxOnMachine_][myMachineIdx_],
          (*padding)[myDeviceIdxOnMachine_],
          myMachineIdx_,
          diffuseComm_,
          diffuseStream_);
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  diffuseToAllGatherEvent.record(diffuseStream_);

  diffuseToAllGatherEvent.block(allGatherStream_);
  if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doAllGather(
        slice3d, myDeviceIdxOnMachine_, allGatherComm_, allGatherStream_);
    if (padding) {
      doAllGather(
          *padding, myDeviceIdxOnMachine_, allGatherComm_, allGatherStream_);
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  if (padding) {
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(slice.data_ptr()) +
            slice3d.numel() * elementSizeInBytes,
        (*padding).data_ptr(),
        (slice.numel() - slice3d.numel()) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        allGatherStream_));
    (*paddingEvent).record(allGatherStream_);
  }
}

void DeviceFairring::reduceScatterOneSlice(
    c10d::ReduceOp opType,
    at::Tensor input,
    at::Tensor output,
    at::cuda::CUDAEvent initialEvent) {
  c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);

  at::cuda::CUDAEvent reduceScatterToCollectEvent;
  at::cuda::CUDAEvent collectToAddEvent;

  MY_CHECK(input.numel() % (numDevicesPerMachine_ * numMachines_) == 0);
  MY_CHECK(
      input.numel() == output.numel() * numDevicesPerMachine_ * numMachines_);
  at::Tensor input3d;
  if (deviceGlobalRankIsFavorable_) {
    input3d = input.view({numDevicesPerMachine_, numMachines_, -1});
  } else {
    input3d =
        input.view({numMachines_, numDevicesPerMachine_, -1}).transpose(0, 1);
  }

  MY_CHECK(numDevicesPerMachine_ >= 2);
  at::Tensor input3dStaging =
      input3d[(myDeviceIdxOnMachine_ + 1) % numDevicesPerMachine_];

  initialEvent.block(reduceScatterStream_);

  if (numMachines_ == 1) {
    MY_CHECK(input3d.is_contiguous());
    NCCL_CHECK(ncclReduceScatter(
        input3d.data_ptr(),
        output.data_ptr(),
        output.numel(),
        torchToNcclDtype(output.scalar_type()),
        torchToNcclRedOp(opType),
        reduceScatterComm_.get(),
        reduceScatterStream_));
  } else if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doReduceScatter(
        input3d,
        myDeviceIdxOnMachine_,
        opType,
        reduceScatterComm_,
        reduceScatterStream_);
    NCCL_CHECK(ncclGroupEnd());
  }
  reduceScatterToCollectEvent.record(reduceScatterStream_);

  reduceScatterToCollectEvent.block(collectStream_);
  if (numMachines_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doCollect(
        input3d[myDeviceIdxOnMachine_],
        input3dStaging,
        myMachineIdx_,
        collectComm_,
        collectStream_);
    NCCL_CHECK(ncclGroupEnd());
  }
  collectToAddEvent.record(collectStream_);

  collectToAddEvent.block(addStream_);
  if (numMachines_ > 1) {
    doReduction(input3dStaging, output, opType, myMachineIdx_, addStream_);
  }
}

void DeviceFairring::allGatherOneSlice(
    at::Tensor input,
    at::Tensor output,
    at::cuda::CUDAEvent initialEvent) {
  c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);

  at::cuda::CUDAEvent diffuseToAllGatherEvent;

  MY_CHECK(output.numel() % (numDevicesPerMachine_ * numMachines_) == 0);
  MY_CHECK(
      output.numel() == input.numel() * numDevicesPerMachine_ * numMachines_);
  at::Tensor output3d;
  if (deviceGlobalRankIsFavorable_) {
    output3d = output.view({numDevicesPerMachine_, numMachines_, -1});
  } else {
    output3d =
        output.view({numMachines_, numDevicesPerMachine_, -1}).transpose(0, 1);
  }

  initialEvent.block(diffuseStream_);

  if (numMachines_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doDiffuse(
        input,
        output3d[myDeviceIdxOnMachine_],
        myMachineIdx_,
        diffuseComm_,
        diffuseStream_);
    NCCL_CHECK(ncclGroupEnd());
  }
  diffuseToAllGatherEvent.record(diffuseStream_);

  diffuseToAllGatherEvent.block(allGatherStream_);
  if (numMachines_ == 1) {
    MY_CHECK(output3d.is_contiguous());
    NCCL_CHECK(ncclAllGather(
        input.data_ptr(),
        output3d.data_ptr(),
        input.numel(),
        torchToNcclDtype(input.scalar_type()),
        allGatherComm_.get(),
        allGatherStream_));
  } else if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    doAllGather(
        output3d, myDeviceIdxOnMachine_, allGatherComm_, allGatherStream_);
    NCCL_CHECK(ncclGroupEnd());
  }
}

} // namespace fairring
