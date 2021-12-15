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
    size_t sliceSizeInBytes)
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
          sliceSizeInBytes,
          numMachines_,
          numDevicesPerMachine_)),
      // Streams
      reduceScatterStream_(CudaStream(myDeviceIdxOnProcess_)),
      collectStream_(CudaStream(myDeviceIdxOnProcess_)),
      addStream_(CudaStream(myDeviceIdxOnProcess_)),
      diffuseStream_(CudaStream(myDeviceIdxOnProcess_)),
      allGatherStream_(CudaStream(myDeviceIdxOnProcess_)),
      // Buffers
      paddingBuffer_(at::empty(
          {static_cast<long>(layout_.numSlots),
           static_cast<long>(numDevicesPerMachine_),
           static_cast<long>(kAlignment)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      diffusedBuffer_(at::empty(
          {static_cast<long>(layout_.numSlots),
           static_cast<long>(layout_.slotSizeInBytes)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      collectedBuffer_(at::empty(
          {static_cast<long>(layout_.numSlots),
           static_cast<long>(numMachines_),
           static_cast<long>(layout_.shardSizeInBytes)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      reducedBuffer_(at::empty(
          {static_cast<long>(layout_.numSlots),
           static_cast<long>(layout_.shardSizeInBytes)},
          c10::TensorOptions()
              .dtype(c10::kByte)
              .device(c10::Device(c10::kCUDA, myDeviceIdxOnProcess_)))),
      // Events
      allgatherToReduceScatterEvents_(makeManyCudaEvents(layout_.numSlots)),
      addToCollectEvents_(makeManyCudaEvents(layout_.numSlots)),
      diffuseToAddEvents_(makeManyCudaEvents(layout_.numSlots)) {
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
            seqNum,
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
    size_t seqNum,
    at::Tensor slice,
    c10::optional<at::cuda::CUDAEvent> initialEvent) {
  c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);

  size_t slotIdx = seqNum % layout_.numSlots;
  size_t sliceSizeInElems = slice.numel();
  c10::ScalarType dtype = slice.scalar_type();
  size_t baseSlotSizeInElems = sliceSizeInElems / numDevicesPerMachine_;
  size_t fullSlotSizeInElems =
      ceilOfRatio(sliceSizeInElems, numDevicesPerMachine_);
  bool sliceNotMultipleOfSlot = sliceSizeInElems % numDevicesPerMachine_ > 0;
  size_t elementSizeInBytes = slice.element_size();
  size_t myShardSizeInBytes = getShardSizeInBytes(
      myMachineIdx_, numMachines_, fullSlotSizeInElems, elementSizeInBytes);
  size_t myShardSizeInElems = myShardSizeInBytes / elementSizeInBytes;

  at::cuda::CUDAEvent reduceScatterToCollectEvent;
  at::cuda::CUDAEvent collectToAddEvent;
  at::cuda::CUDAEvent addToDiffuseEvent;
  at::cuda::CUDAEvent diffuseToAllGatherEvent;

  if (initialEvent.has_value()) {
    initialEvent.value().block(reduceScatterStream_);
  }
  allgatherToReduceScatterEvents_[slotIdx].block(reduceScatterStream_);
  NCCL_CHECK(ncclReduceScatter(
      slice.data_ptr(),
      diffusedBuffer_[slotIdx].data_ptr(),
      baseSlotSizeInElems,
      torchToNcclDtype(dtype),
      ncclSum,
      reduceScatterComm_.get(),
      reduceScatterStream_));
  if (sliceNotMultipleOfSlot) {
    // No need to zero out the padding: we don't care what value it has/gets.
    CUDA_CHECK(cudaMemcpyAsync(
        paddingBuffer_[slotIdx].data_ptr(),
        reinterpret_cast<uint8_t*>(slice.data_ptr()) +
            baseSlotSizeInElems * numDevicesPerMachine_ * elementSizeInBytes,
        (sliceSizeInElems % numDevicesPerMachine_) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        reduceScatterStream_));
    NCCL_CHECK(ncclReduceScatter(
        paddingBuffer_[slotIdx].data_ptr(),
        reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
            baseSlotSizeInElems * elementSizeInBytes,
        1,
        torchToNcclDtype(dtype),
        ncclSum,
        reduceScatterComm_.get(),
        reduceScatterStream_));
  }
  reduceScatterToCollectEvent.record(reduceScatterStream_);

  reduceScatterToCollectEvent.block(collectStream_);
  addToCollectEvents_[slotIdx].block(collectStream_);
  NCCL_CHECK(ncclGroupStart());
  for (const auto serverMachineIdx : c10::irange(numMachines_)) {
    void* slicePtr =
        reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
        getShardOffsetInBytes(
            serverMachineIdx,
            numMachines_,
            fullSlotSizeInElems,
            elementSizeInBytes);
    size_t sliceSizeInBytes = getShardSizeInBytes(
        serverMachineIdx,
        numMachines_,
        fullSlotSizeInElems,
        elementSizeInBytes);
    NCCL_CHECK(ncclSend(
        slicePtr,
        sliceSizeInBytes / elementSizeInBytes,
        torchToNcclDtype(dtype),
        serverMachineIdx,
        collectComm_.get(),
        collectStream_));
  }
  for (const auto clientMachineIdx : c10::irange(numMachines_)) {
    void* slicePtr = collectedBuffer_
                         .index(
                             {static_cast<long>(slotIdx),
                              static_cast<long>(clientMachineIdx)})
                         .data_ptr();
    NCCL_CHECK(ncclRecv(
        slicePtr,
        myShardSizeInElems,
        torchToNcclDtype(dtype),
        clientMachineIdx,
        collectComm_.get(),
        collectStream_));
  }
  NCCL_CHECK(ncclGroupEnd());
  collectToAddEvent.record(collectStream_);

  collectToAddEvent.block(addStream_);
  diffuseToAddEvents_[slotIdx].block(addStream_);
  {
    c10::cuda::CUDAStreamGuard g(addStream_);
    // sum_out wants its first argument to be an lvalue (for no good reason)
    auto out = reducedBuffer_.view(dtype).index(
        {static_cast<long>(slotIdx),
         torch::indexing::Slice(0, myShardSizeInElems)});
    at::sum_out(
        out,
        collectedBuffer_.view(dtype).index(
            {static_cast<long>(slotIdx),
             torch::indexing::Slice(),
             torch::indexing::Slice(0, myShardSizeInElems)}),
        {0});
  }
  addToCollectEvents_[slotIdx].record(addStream_);
  addToDiffuseEvent.record(addStream_);

  addToDiffuseEvent.block(diffuseStream_);
  NCCL_CHECK(ncclGroupStart());
  for (const auto clientMachineIdx : c10::irange(numMachines_)) {
    NCCL_CHECK(ncclSend(
        reducedBuffer_[slotIdx].data_ptr(),
        myShardSizeInElems,
        torchToNcclDtype(dtype),
        clientMachineIdx,
        diffuseComm_.get(),
        diffuseStream_));
  }
  for (const auto serverMachineIdx : c10::irange(numMachines_)) {
    void* slicePtr =
        reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
        getShardOffsetInBytes(
            serverMachineIdx,
            numMachines_,
            fullSlotSizeInElems,
            elementSizeInBytes);
    size_t sliceSizeInBytes = getShardSizeInBytes(
        serverMachineIdx,
        numMachines_,
        fullSlotSizeInElems,
        elementSizeInBytes);
    NCCL_CHECK(ncclRecv(
        slicePtr,
        sliceSizeInBytes / elementSizeInBytes,
        torchToNcclDtype(dtype),
        serverMachineIdx,
        diffuseComm_.get(),
        diffuseStream_));
  }
  NCCL_CHECK(ncclGroupEnd());
  diffuseToAddEvents_[slotIdx].record(diffuseStream_);
  diffuseToAllGatherEvent.record(diffuseStream_);

  diffuseToAllGatherEvent.block(allGatherStream_);
  NCCL_CHECK(ncclAllGather(
      diffusedBuffer_[slotIdx].data_ptr(),
      slice.data_ptr(),
      baseSlotSizeInElems,
      torchToNcclDtype(dtype),
      allGatherComm_.get(),
      allGatherStream_));
  if (sliceNotMultipleOfSlot) {
    NCCL_CHECK(ncclAllGather(
        reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
            baseSlotSizeInElems * elementSizeInBytes,
        paddingBuffer_[slotIdx].data_ptr(),
        1,
        torchToNcclDtype(dtype),
        allGatherComm_.get(),
        allGatherStream_));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(slice.data_ptr()) +
            baseSlotSizeInElems * numDevicesPerMachine_ * elementSizeInBytes,
        paddingBuffer_[slotIdx].data_ptr(),
        (sliceSizeInElems % numDevicesPerMachine_) * elementSizeInBytes,
        cudaMemcpyDeviceToDevice,
        allGatherStream_));
  }
  allgatherToReduceScatterEvents_[slotIdx].record(allGatherStream_);
}

} // namespace fairring
