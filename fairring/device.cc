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

std::vector<CudaStream> makeManyCudaStreams(size_t amount, int deviceIdx) {
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

} // namespace

DeviceFairring::DeviceFairring(
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
    size_t maxPaddingAllocatedInBytes,
    size_t minParallelism)
    : // Arguments
      myDeviceIdxOnProcess_(deviceIdxOnProcess),
      myMachineIdx_(machineIdx),
      myDeviceIdxOnMachine_(deviceIdxOnMachine),
      numMachines_(numMachines),
      numDevicesPerMachine_(numDevicesPerMachine),
      store_(std::move(store)),
      reduceScatterComm_(std::move(reduceScatterComm)),
      collectComm_(std::move(collectComm)),
      diffuseComm_(std::move(diffuseComm)),
      allGatherComm_(std::move(allGatherComm)),
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
          {static_cast<long>(layout_.numPaddingSlots),
           static_cast<long>(numDevicesPerMachine_),
           static_cast<long>(numMachines_),
           static_cast<long>(kAlignment)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      paddingEvents_(makeManyCudaEvents(layout_.numPaddingSlots)),
      // Staging
      stagingBuffer_(at::empty(
          {static_cast<long>(layout_.numStagingSlots),
           static_cast<long>(numMachines_),
           static_cast<long>(layout_.slotSizeInBytes)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      paddingStagingBuffer_(at::empty(
          {static_cast<long>(layout_.numStagingSlots),
           static_cast<long>(numMachines_),
           static_cast<long>(kAlignment)},
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
                     tensor = std::move(tensor),
                     initialEvent = std::make_shared<at::cuda::CUDAEvent>(
                         std::move(initialEvent)),
                     future]() mutable {
    size_t numElements = tensor.numel();
    MY_CHECK(kAlignment % tensor.element_size() == 0);
    size_t maxSliceSizeInElems =
        layout_.sliceSizeInBytes / tensor.element_size();
    size_t numSlices = ceilOfRatio(numElements, maxSliceSizeInElems);
    for (const auto sliceIdx : c10::irange(numSlices)) {
      size_t seqNum = nextSlot_++;
      size_t offsetInElems = sliceIdx * maxSliceSizeInElems;
      size_t sliceSizeInElems =
          std::min(maxSliceSizeInElems, numElements - offsetInElems);
      at::Tensor slice = tensor.flatten().slice(
          /*dim=*/0, offsetInElems, offsetInElems + sliceSizeInElems);
      auto myFuture = sliceIdx == numSlices - 1
          ? std::move(future)
          : c10::intrusive_ptr<c10::ivalue::Future>();
      try {
        processOneSlice(
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

void DeviceFairring::processOneSlice(
    at::Tensor slice,
    c10::optional<at::cuda::CUDAEvent> initialEvent) {
  c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);

  c10::ScalarType dtype = slice.scalar_type();
  size_t elementSizeInBytes = slice.element_size();

  at::cuda::CUDAEvent reduceScatterToCollectEvent;
  at::cuda::CUDAEvent collectToAddEvent;
  at::cuda::CUDAEvent addToDiffuseEvent;
  at::cuda::CUDAEvent diffuseToAllGatherEvent;

  at::Tensor slice3d;
  c10::optional<at::Tensor> padding;
  at::cuda::CUDAEvent* paddingEvent = nullptr;
  if (slice.numel() % (numDevicesPerMachine_ * numMachines_) == 0) {
    slice3d = slice.view(
        {static_cast<long>(numDevicesPerMachine_),
         static_cast<long>(numMachines_),
         -1});
  } else {
    size_t sliceSizeInElems = roundDownToNearestMultiple(
        static_cast<size_t>(slice.numel()),
        numDevicesPerMachine_ * numMachines_);
    slice3d = slice.index({torch::indexing::Slice(0, sliceSizeInElems)})
                  .view(
                      {static_cast<long>(numDevicesPerMachine_),
                       static_cast<long>(numMachines_),
                       -1});
    size_t paddingSlotIdx = (nextPaddingSlot_++) % layout_.numPaddingSlots;
    padding = paddingBuffer_[paddingSlotIdx]
                  .view(dtype)
                  .flatten()
                  .index({torch::indexing::Slice(
                      0, numDevicesPerMachine_ * numMachines_)})
                  .view(
                      {static_cast<long>(numDevicesPerMachine_),
                       static_cast<long>(numMachines_)});
    paddingEvent = &paddingEvents_[paddingSlotIdx];
  }

  at::Tensor slice3dStaging;
  c10::optional<at::Tensor> paddingStaging;
  at::cuda::CUDAEvent* stagingEvent = nullptr;
  if (numDevicesPerMachine_ == 1) {
    size_t stagingSlotIdx = (nextStagingSlot_++) % layout_.numStagingSlots;
    slice3dStaging = stagingBuffer_[stagingSlotIdx]
                         .view(dtype)
                         .flatten()
                         .index({torch::indexing::Slice(
                             0, slice3d[myDeviceIdxOnMachine_].numel())})
                         .view({static_cast<long>(numMachines_), -1});
    if (padding) {
      paddingStaging = paddingStagingBuffer_[stagingSlotIdx]
                           .view(dtype)
                           .flatten()
                           .index({torch::indexing::Slice(
                               0, (*padding)[myDeviceIdxOnMachine_].numel())})
                           .view({static_cast<long>(numMachines_)});
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
        (slice.numel() % slice3d.numel()) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        reduceScatterStream_));
  }

  if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclReduceScatter(
        slice3d.data_ptr(),
        slice3d[myDeviceIdxOnMachine_].data_ptr(),
        slice3d[myDeviceIdxOnMachine_].numel(),
        torchToNcclDtype(dtype),
        ncclSum,
        reduceScatterComm_.get(),
        reduceScatterStream_));
    if (padding) {
      NCCL_CHECK(ncclReduceScatter(
          (*padding).data_ptr(),
          (*padding)[myDeviceIdxOnMachine_].data_ptr(),
          (*padding)[myDeviceIdxOnMachine_].numel(),
          torchToNcclDtype(dtype),
          ncclSum,
          reduceScatterComm_.get(),
          reduceScatterStream_));
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
    for (const auto serverMachineIdx : c10::irange(numMachines_)) {
      NCCL_CHECK(ncclSend(
          slice3d[myDeviceIdxOnMachine_][serverMachineIdx].data_ptr(),
          slice3d[myDeviceIdxOnMachine_][serverMachineIdx].numel(),
          torchToNcclDtype(dtype),
          serverMachineIdx,
          collectComm_.get(),
          collectStream_));
    }
    for (const auto clientMachineIdx : c10::irange(numMachines_)) {
      NCCL_CHECK(ncclRecv(
          slice3dStaging[clientMachineIdx].data_ptr(),
          slice3dStaging[clientMachineIdx].numel(),
          torchToNcclDtype(dtype),
          clientMachineIdx,
          collectComm_.get(),
          collectStream_));
    }
    if (padding) {
      for (const auto serverMachineIdx : c10::irange(numMachines_)) {
        NCCL_CHECK(ncclSend(
            (*padding)[myDeviceIdxOnMachine_][serverMachineIdx].data_ptr(),
            (*padding)[myDeviceIdxOnMachine_][serverMachineIdx].numel(),
            torchToNcclDtype(dtype),
            serverMachineIdx,
            collectComm_.get(),
            collectStream_));
      }
      for (const auto clientMachineIdx : c10::irange(numMachines_)) {
        NCCL_CHECK(ncclRecv(
            (*paddingStaging)[clientMachineIdx].data_ptr(),
            (*paddingStaging)[clientMachineIdx].numel(),
            torchToNcclDtype(dtype),
            clientMachineIdx,
            collectComm_.get(),
            collectStream_));
      }
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  collectToAddEvent.record(collectStream_);

  collectToAddEvent.block(addStream_);
  if (numMachines_ > 1) {
    c10::cuda::CUDAStreamGuard g(addStream_);
    // sum_out wants its first argument to be an lvalue (for no good reason)
    auto out = slice3d[myDeviceIdxOnMachine_][myMachineIdx_];
    at::sum_out(out, slice3dStaging, {0});
    if (padding) {
      auto paddingOut = (*padding)[myDeviceIdxOnMachine_][myMachineIdx_];
      at::sum_out(paddingOut, (*paddingStaging), {0});
    }
  }
  addToDiffuseEvent.record(addStream_);

  if (stagingEvent) {
    (*stagingEvent).record(addStream_);
  }

  addToDiffuseEvent.block(diffuseStream_);
  if (numMachines_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    for (const auto clientMachineIdx : c10::irange(numMachines_)) {
      if (clientMachineIdx != myMachineIdx_) {
        NCCL_CHECK(ncclSend(
            slice3d[myDeviceIdxOnMachine_][myMachineIdx_].data_ptr(),
            slice3d[myDeviceIdxOnMachine_][myMachineIdx_].numel(),
            torchToNcclDtype(dtype),
            clientMachineIdx,
            diffuseComm_.get(),
            diffuseStream_));
      }
    }
    for (const auto serverMachineIdx : c10::irange(numMachines_)) {
      if (serverMachineIdx != myMachineIdx_) {
        NCCL_CHECK(ncclRecv(
            slice3d[myDeviceIdxOnMachine_][serverMachineIdx].data_ptr(),
            slice3d[myDeviceIdxOnMachine_][serverMachineIdx].numel(),
            torchToNcclDtype(dtype),
            serverMachineIdx,
            diffuseComm_.get(),
            diffuseStream_));
      }
    }
    if (padding) {
      for (const auto clientMachineIdx : c10::irange(numMachines_)) {
        if (clientMachineIdx != myMachineIdx_) {
          NCCL_CHECK(ncclSend(
              (*padding)[myDeviceIdxOnMachine_][myMachineIdx_].data_ptr(),
              (*padding)[myDeviceIdxOnMachine_][myMachineIdx_].numel(),
              torchToNcclDtype(dtype),
              clientMachineIdx,
              diffuseComm_.get(),
              diffuseStream_));
        }
      }
      for (const auto serverMachineIdx : c10::irange(numMachines_)) {
        if (serverMachineIdx != myMachineIdx_) {
          NCCL_CHECK(ncclRecv(
              (*padding)[myDeviceIdxOnMachine_][serverMachineIdx].data_ptr(),
              (*padding)[myDeviceIdxOnMachine_][serverMachineIdx].numel(),
              torchToNcclDtype(dtype),
              serverMachineIdx,
              diffuseComm_.get(),
              diffuseStream_));
        }
      }
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  diffuseToAllGatherEvent.record(diffuseStream_);

  diffuseToAllGatherEvent.block(allGatherStream_);
  if (numDevicesPerMachine_ > 1) {
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllGather(
        slice3d[myDeviceIdxOnMachine_].data_ptr(),
        slice3d.data_ptr(),
        slice3d[myDeviceIdxOnMachine_].numel(),
        torchToNcclDtype(dtype),
        allGatherComm_.get(),
        allGatherStream_));
    if (padding) {
      NCCL_CHECK(ncclAllGather(
          (*padding)[myDeviceIdxOnMachine_].data_ptr(),
          (*padding).data_ptr(),
          (*padding)[myDeviceIdxOnMachine_].numel(),
          torchToNcclDtype(dtype),
          allGatherComm_.get(),
          allGatherStream_));
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  if (padding) {
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(slice.data_ptr()) +
            slice3d.numel() * elementSizeInBytes,
        (*padding).data_ptr(),
        (slice.numel() % slice3d.numel()) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        allGatherStream_));
    (*paddingEvent).block(reduceScatterStream_);
  }
}

} // namespace fairring
