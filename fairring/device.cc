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
#include <tensorpipe/tensorpipe_cuda.h>

#include <fairring/tpcoro.h>

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
      allGatherComm_(std::move(allGatherComm)),
      layout_(computeLayout(
          maxMemoryAllocatedInBytes,
          sliceSizeInBytes,
          numMachines_,
          numDevicesPerMachine_)),
      // TensorPipe
      tpCtx_(createTpContext(
          std::to_string(myMachineIdx_) + "." +
          std::to_string(myDeviceIdxOnMachine_))),
      // Sequencers
      slotSequencers_(layout_.numSlots),
      leafTpWriteSequencers_(numMachines_),
      rootTpReadDescriptorSequencers_(numMachines_),
      rootTpReadSequencers_(numMachines_),
      rootTpWriteSequencers_(numMachines_),
      leafTpReadDescriptorSequencers_(numMachines_),
      leafTpReadSequencers_(numMachines_),
      // Streams
      reduceScatterStream_(CudaStream(myDeviceIdxOnProcess_)),
      leafCollectStreams_(
          makeManyCudaStreams(numMachines_, myDeviceIdxOnProcess_)),
      rootCollectStreams_(
          makeManyCudaStreams(numMachines_, myDeviceIdxOnProcess_)),
      addStream_(CudaStream(myDeviceIdxOnProcess_)),
      rootDiffuseStreams_(
          makeManyCudaStreams(numMachines_, myDeviceIdxOnProcess_)),
      leafDiffuseStreams_(
          makeManyCudaStreams(numMachines_, myDeviceIdxOnProcess_)),
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
      addToCollectEvents_(makeManyCudaEvents(layout_.numSlots, numMachines_)),
      diffuseToAddEvents_(makeManyCudaEvents(layout_.numSlots, numMachines_)) {
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

void DeviceFairring::startListening() {
  tensorpipe::Error error;
  std::string address;
  const char* iface = std::getenv("TP_SOCKET_IFNAME");
  std::tie(error, address) = iface != nullptr
      ? tensorpipe::transport::uv::lookupAddrForIface(std::string(iface))
      : tensorpipe::transport::uv::lookupAddrLikeNccl();
  if (error) {
    LOG(ERROR) << error.what();
    throw std::runtime_error(error.what());
  }
  tpListener_ = tpCtx_->listen({"ibv://" + address});
  std::string key = "machines/" + std::to_string(myMachineIdx_) + "/" +
      std::to_string(myDeviceIdxOnMachine_) + "/address";
  store_->set(key, stringToByteVector(tpListener_->url("ibv")));

  key = "machines/" + std::to_string(myMachineIdx_) + "/" +
      std::to_string(myDeviceIdxOnMachine_) + "/device_idx";
  store_->set(key, integerToByteVector(myDeviceIdxOnProcess_));
}

void DeviceFairring::connect() {
  leafTpPipes_.reserve(numMachines_);
  for (const auto serverMachineIdx : c10::irange(numMachines_)) {
    std::string key = "machines/" + std::to_string(serverMachineIdx) + "/" +
        std::to_string(myDeviceIdxOnMachine_) + "/address";
    std::string addr = byteVectorToString(store_->get(key));
    leafTpPipes_.push_back(tpCtx_->connect(
        addr,
        tensorpipe::PipeOptions().remoteName(
            std::to_string(serverMachineIdx) + "." +
            std::to_string(myDeviceIdxOnMachine_))));
  }

  remoteDeviceIndices_.reserve(numMachines_);
  for (const auto otherMachineIdx : c10::irange(numMachines_)) {
    std::string key = "machines/" + std::to_string(otherMachineIdx) + "/" +
        std::to_string(myDeviceIdxOnMachine_) + "/device_idx";
    int64_t deviceIdx = byteVectorToInteger(store_->get(key));
    remoteDeviceIndices_.push_back(deviceIdx);
  }
}

void DeviceFairring::acceptPipes() {
  rootTpPipes_.resize(numMachines_);
  for (const auto clientMachineIdx : c10::irange(numMachines_)) {
    std::promise<std::shared_ptr<tensorpipe::Pipe>> pipePromise;
    tpListener_->accept([&](const tensorpipe::Error& error,
                            std::shared_ptr<tensorpipe::Pipe> pipe) {
      if (error) {
        LOG(ERROR) << error.what();
        pipePromise.set_exception(
            std::make_exception_ptr(std::runtime_error(error.what())));
        return;
      }
      pipePromise.set_value(std::move(pipe));
    });
    std::shared_ptr<tensorpipe::Pipe> pipe = pipePromise.get_future().get();
    size_t actualClientMachineIdx = std::strtoull(
        pipe->getRemoteName().c_str(), /*end=*/nullptr, /*base=*/10);
    rootTpPipes_[actualClientMachineIdx] = std::move(pipe);
  }
}

DeviceFairring::~DeviceFairring() {
  cmdQueue_.enqueue(nullptr);
  cmdThread_.join();
  tracker_.waitForAllCoros();
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
      tpcoro::Task task = tpcoro::co_invoke(
          [this,
           seqNum,
           initialEvent = sliceIdx == 0
               ? c10::optional<at::cuda::CUDAEvent>(std::move(*initialEvent))
               : c10::nullopt,
           slice = std::move(slice),
           tensor,
           future = sliceIdx == numSlices - 1
               ? std::move(future)
               : c10::intrusive_ptr<c10::ivalue::Future>()]() mutable
          -> tpcoro::Task {
            try {
              co_await processOneSlice(
                  seqNum, std::move(slice), std::move(initialEvent));
              if (future) {
                c10::cuda::CUDAStreamGuard g(allGatherStream_);
                future->markCompleted(
                    std::vector<at::Tensor>{std::move(tensor)});
              }
            } catch (const std::exception& e) {
              LOG(ERROR) << "Coroutine for chunk #" << seqNum
                         << " threw exception: " << e.what();
              if (future) {
                c10::cuda::CUDAStreamGuard g(allGatherStream_);
                future->setError(std::current_exception());
              }
            }
          });
      // Leave task unawaited: it's eager, it's gonna be fine.
    }
  });

  return future;
}

tpcoro::Task DeviceFairring::processOneSlice(
    size_t seqNum,
    at::Tensor slice,
    c10::optional<at::cuda::CUDAEvent> initialEvent) {
  size_t slotIdx = seqNum % layout_.numSlots;
  size_t seqNumForSlot = seqNum / layout_.numSlots;
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

  tpcoro::CoroTracker::Beacon beacon = tracker_.startTrackingMe();

  at::cuda::CUDAEvent reduceScatterToCollectEvent;
  std::vector<at::cuda::CUDAEvent> collectToDiffuseEvents(numMachines_);
  std::vector<at::cuda::CUDAEvent> collectToAddEvents(numMachines_);
  std::vector<at::cuda::CUDAEvent> addToDiffuseEvents(numMachines_);
  std::vector<at::cuda::CUDAEvent> diffuseToAllGatherEvents(numMachines_);

  tpcoro::Sequencer::TurnToken slotTT =
      co_await slotSequencers_[slotIdx].waitForTurn(seqNumForSlot);

  tpcoro::Sequencer::TurnToken reduceScatterTT =
      co_await reduceScatterSequencer_.waitForTurn(seqNum);
  if (initialEvent.has_value()) {
    initialEvent.value().block(reduceScatterStream_);
  }
  allgatherToReduceScatterEvents_[slotIdx].block(reduceScatterStream_);
  {
    c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);
    NCCL_CHECK(ncclReduceScatter(
        slice.data_ptr(),
        diffusedBuffer_[slotIdx].data_ptr(),
        baseSlotSizeInElems,
        torchToNcclDtype(slice.scalar_type()),
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
          torchToNcclDtype(slice.scalar_type()),
          ncclSum,
          reduceScatterComm_.get(),
          reduceScatterStream_));
    }
  }
  reduceScatterToCollectEvent.record(reduceScatterStream_);
  reduceScatterTT.release();

  std::vector<tpcoro::Task> leafCollectTasks = tpcoro::forIdxInRange(
      numMachines_, [&, this](size_t serverMachineIdx) -> tpcoro::Task {
        tpcoro::Sequencer::TurnToken writeTT =
            co_await leafTpWriteSequencers_[serverMachineIdx].waitForTurn(
                seqNum);
        reduceScatterToCollectEvent.block(
            leafCollectStreams_[serverMachineIdx]);
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
        tensorpipe::Message message;
        message.tensors.resize(1);
        message.tensors[0] = {
            .buffer =
                tensorpipe::CudaBuffer{
                    .ptr = slicePtr,
                    .stream = leafCollectStreams_[serverMachineIdx],
                },
            .length = sliceSizeInBytes,
            .targetDevice = tensorpipe::Device(
                tensorpipe::kCudaDeviceType,
                remoteDeviceIndices_[serverMachineIdx]),
        };
        auto writeAwaitable = tpcoro::TpWrite(
            *leafTpPipes_[serverMachineIdx], std::move(message));
        writeTT.release();
        co_await writeAwaitable;
      });

  std::vector<tpcoro::Task> rootCollectTasks = tpcoro::forIdxInRange(
      numMachines_, [&, this](size_t clientMachineIdx) -> tpcoro::Task {
        tpcoro::Sequencer::TurnToken readDescTT =
            co_await rootTpReadDescriptorSequencers_[clientMachineIdx]
                .waitForTurn(seqNum);
        auto readDescAwaitable =
            tpcoro::TpReadDescriptor(*rootTpPipes_[clientMachineIdx]);
        readDescTT.release();
        co_await readDescAwaitable;
        tensorpipe::Descriptor descriptor =
            std::move(readDescAwaitable).getDescriptor();
        // FIXME Check the descriptor is as expected?
        tpcoro::Sequencer::TurnToken readTT =
            co_await rootTpReadSequencers_[clientMachineIdx].waitForTurn(
                seqNum);
        addToCollectEvents_[slotIdx][clientMachineIdx].block(
            rootCollectStreams_[clientMachineIdx]);
        tensorpipe::Allocation allocation;
        allocation.tensors.resize(1);
        allocation.tensors[0].buffer = tensorpipe::CudaBuffer{
            .ptr = collectedBuffer_
                       .index(
                           {static_cast<long>(slotIdx),
                            static_cast<long>(clientMachineIdx)})
                       .data_ptr(),
            .stream = rootCollectStreams_[clientMachineIdx],
        };
        auto readAwaitable = tpcoro::TpRead(
            *rootTpPipes_[clientMachineIdx], std::move(allocation));
        readTT.release();
        co_await readAwaitable;
        collectToAddEvents[clientMachineIdx].record(
            rootCollectStreams_[clientMachineIdx]);
      });

  // No need to wait for the leaf tasks to complete in order to advance.
  co_await tpcoro::Parallel(std::move(rootCollectTasks));

  for (const auto clientMachineIdx : c10::irange(numMachines_)) {
    diffuseToAddEvents_[slotIdx][clientMachineIdx].block(addStream_);
    collectToAddEvents[clientMachineIdx].block(addStream_);
  }
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
  for (const auto clientMachineIdx : c10::irange(numMachines_)) {
    addToDiffuseEvents[clientMachineIdx].record(addStream_);
    addToCollectEvents_[slotIdx][clientMachineIdx].record(addStream_);
  }

  std::vector<tpcoro::Task> rootDiffuseTasks = tpcoro::forIdxInRange(
      numMachines_, [&, this](size_t clientMachineIdx) -> tpcoro::Task {
        tpcoro::Sequencer::TurnToken writeTT =
            co_await rootTpWriteSequencers_[clientMachineIdx].waitForTurn(
                seqNum);
        for (const auto additionMachineIdx : c10::irange(numMachines_)) {
          addToDiffuseEvents[additionMachineIdx].block(
              rootDiffuseStreams_[clientMachineIdx]);
        }
        tensorpipe::Message message;
        message.tensors.resize(1);
        message.tensors[0] = {
            .buffer =
                tensorpipe::CudaBuffer{
                    .ptr = reducedBuffer_[slotIdx].data_ptr(),
                    .stream = rootDiffuseStreams_[clientMachineIdx],
                },
            .length = myShardSizeInBytes,
            .targetDevice = tensorpipe::Device(
                tensorpipe::kCudaDeviceType,
                remoteDeviceIndices_[clientMachineIdx]),
        };
        auto writeAwaitable = tpcoro::TpWrite(
            *rootTpPipes_[clientMachineIdx], std::move(message));
        writeTT.release();
        co_await writeAwaitable;
        diffuseToAddEvents_[slotIdx][clientMachineIdx].record(
            rootDiffuseStreams_[clientMachineIdx]);
      });

  std::vector<tpcoro::Task> leafDiffuseTasks = tpcoro::forIdxInRange(
      numMachines_, [&, this](size_t serverMachineIdx) -> tpcoro::Task {
        // The write must be done for the "diffused" buffer to be safe to reuse.
        co_await leafCollectTasks[serverMachineIdx];

        tpcoro::Sequencer::TurnToken readDescTT =
            co_await leafTpReadDescriptorSequencers_[serverMachineIdx]
                .waitForTurn(seqNum);
        collectToDiffuseEvents[serverMachineIdx].record(
            leafCollectStreams_[serverMachineIdx]);
        auto readDescAwaitable =
            tpcoro::TpReadDescriptor(*leafTpPipes_[serverMachineIdx]);
        readDescTT.release();
        co_await readDescAwaitable;
        tensorpipe::Descriptor descriptor =
            std::move(readDescAwaitable).getDescriptor();
        // FIXME Check the descriptor is as expected?
        tpcoro::Sequencer::TurnToken readTT =
            co_await leafTpReadSequencers_[serverMachineIdx].waitForTurn(
                seqNum);
        collectToDiffuseEvents[serverMachineIdx].block(
            leafDiffuseStreams_[serverMachineIdx]);
        void* slicePtr =
            reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
            getShardOffsetInBytes(
                serverMachineIdx,
                numMachines_,
                fullSlotSizeInElems,
                elementSizeInBytes);
        tensorpipe::Allocation allocation;
        allocation.tensors.resize(1);
        allocation.tensors[0].buffer = tensorpipe::CudaBuffer{
            .ptr = slicePtr,
            .stream = leafDiffuseStreams_[serverMachineIdx],
        };
        auto readAwaitable = tpcoro::TpRead(
            *leafTpPipes_[serverMachineIdx], std::move(allocation));
        readTT.release();
        co_await readAwaitable;
        diffuseToAllGatherEvents[serverMachineIdx].record(
            leafDiffuseStreams_[serverMachineIdx]);
      });

  // No need to wait for the root tasks to complete in order to advance.
  co_await tpcoro::Parallel(std::move(leafDiffuseTasks));

  tpcoro::Sequencer::TurnToken allGatherTT =
      co_await allGatherSequencer_.waitForTurn(seqNum);
  for (at::cuda::CUDAEvent& ev : diffuseToAllGatherEvents) {
    ev.block(allGatherStream_);
  }
  {
    c10::cuda::CUDAGuard g(myDeviceIdxOnProcess_);
    NCCL_CHECK(ncclAllGather(
        diffusedBuffer_[slotIdx].data_ptr(),
        slice.data_ptr(),
        baseSlotSizeInElems,
        torchToNcclDtype(slice.scalar_type()),
        allGatherComm_.get(),
        allGatherStream_));
    if (sliceNotMultipleOfSlot) {
      NCCL_CHECK(ncclAllGather(
          reinterpret_cast<uint8_t*>(diffusedBuffer_[slotIdx].data_ptr()) +
              baseSlotSizeInElems * elementSizeInBytes,
          paddingBuffer_[slotIdx].data_ptr(),
          1,
          torchToNcclDtype(slice.scalar_type()),
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
  }
  allgatherToReduceScatterEvents_[slotIdx].record(allGatherStream_);
  allGatherTT.release();

  co_await tpcoro::Parallel(std::move(rootDiffuseTasks));
}

} // namespace fairring
