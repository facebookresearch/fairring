/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fairring/machine.h>

#include <chrono>
#include <fstream>
#include <thread>

#include <sys/types.h>
#include <unistd.h>

#include <c10/cuda/CUDAFunctions.h>

#include <fairring/device.h>
#include <fairring/utils.h>

namespace {

c10::optional<std::string> getBootID() {
  std::ifstream f{"/proc/sys/kernel/random/boot_id"};
  if (!f.is_open()) {
    return c10::nullopt;
  }
  std::string v;
  getline(f, v);
  f.close();
  return v;
}

} // namespace

namespace fairring {

MachineFairring::MachineFairring(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    std::vector<c10::Device> devices,
    size_t maxMemoryAllocatedInBytes,
    size_t maxPaddingAllocatedInBytes,
    size_t minParallelism)
    : devices_(std::move(devices)) {
  TORCH_CHECK(0 <= rank && rank < size);
  std::string machineId = getBootID().value();
  int64_t numDevices = devices_.size();
  TORCH_CHECK(numDevices > 0);
  store->set(
      "rdv/" + std::to_string(rank) + "/machine_idx",
      stringToByteVector(machineId));
  store->set(
      "rdv/" + std::to_string(rank) + "/num_devices",
      integerToByteVector(numDevices));

  struct Machine {
    size_t idx = 0;
    size_t totalDevices = 0;
  };
  std::unordered_map<std::string, Machine> idToMachine;
  struct Device {
    size_t machineIdx = 0;
    size_t idxWithinMachine = 0;
  };
  std::vector<Device> allDevices;
  int64_t idxOfMyMachine = -1;
  int64_t idxOfMyFirstDevice = -1;
  for (const auto otherRank : c10::irange(size)) {
    std::string otherMachineId = byteVectorToString(
        store->get("rdv/" + std::to_string(otherRank) + "/machine_idx"));
    int64_t otherNumDevices = byteVectorToInteger(
        store->get("rdv/" + std::to_string(otherRank) + "/num_devices"));
    auto iter = idToMachine.find(otherMachineId);
    if (iter == idToMachine.end()) {
      size_t newMachineIdx = idToMachine.size();
      std::tie(iter, std::ignore) =
          idToMachine.emplace(otherMachineId, Machine{.idx = newMachineIdx});
    }
    Machine& otherMachine = iter->second;
    if (otherRank == rank) {
      idxOfMyMachine = otherMachine.idx;
      idxOfMyFirstDevice = otherMachine.totalDevices;
    }
    for (const auto deviceOffset : c10::irange(otherNumDevices)) {
      allDevices.push_back(Device{
          .machineIdx = otherMachine.idx,
          .idxWithinMachine = otherMachine.totalDevices + deviceOffset});
    }
    otherMachine.totalDevices += otherNumDevices;
  }

  TORCH_CHECK(!idToMachine.empty());
  TORCH_CHECK(idxOfMyMachine >= 0);
  TORCH_CHECK(idxOfMyFirstDevice >= 0);
  size_t numMachines = idToMachine.size();
  size_t devicesPerMachine = idToMachine.begin()->second.totalDevices;
  for (const auto& kv : idToMachine) {
    TORCH_CHECK(kv.second.totalDevices == devicesPerMachine);
  }

  at::Tensor deviceGlobalRankAssignment = at::empty(
      {static_cast<long>(numMachines), static_cast<long>(devicesPerMachine)},
      at::kLong);
  for (const auto deviceGlobalRank :
       c10::irange(numMachines * devicesPerMachine)) {
    const Device& device = allDevices[deviceGlobalRank];
    deviceGlobalRankAssignment[device.machineIdx][device.idxWithinMachine] =
        static_cast<long>(deviceGlobalRank);
  }
  bool deviceGlobalRankIsFavorable;
  if (deviceGlobalRankAssignment.equal(
          torch::arange(static_cast<long>(numMachines * devicesPerMachine))
              .view(
                  {static_cast<long>(devicesPerMachine),
                   static_cast<long>(numMachines)})
              .transpose(0, 1))) {
    deviceGlobalRankIsFavorable = true;
  } else if (deviceGlobalRankAssignment.equal(
                 torch::arange(
                     static_cast<long>(numMachines * devicesPerMachine))
                     .view(
                         {static_cast<long>(numMachines),
                          static_cast<long>(devicesPerMachine)}))) {
    deviceGlobalRankIsFavorable = false;
  } else {
    MY_CHECK(false);
  }

  ncclUniqueId reduceScatterUniqueId;
  ncclUniqueId allGatherUniqueId;
  if (idxOfMyFirstDevice == 0) {
    NCCL_CHECK(ncclGetUniqueId(&reduceScatterUniqueId));
    NCCL_CHECK(ncclGetUniqueId(&allGatherUniqueId));
    store->set(
        "machines/" + std::to_string(idxOfMyMachine) +
            "/reduce_scatter_nccl_id",
        podToByteString(reduceScatterUniqueId));
    store->set(
        "machines/" + std::to_string(idxOfMyMachine) + "/all_gather_nccl_id",
        podToByteString(allGatherUniqueId));
  } else {
    reduceScatterUniqueId = byteStringToPod<ncclUniqueId>(store->get(
        "machines/" + std::to_string(idxOfMyMachine) +
        "/reduce_scatter_nccl_id"));
    allGatherUniqueId = byteStringToPod<ncclUniqueId>(store->get(
        "machines/" + std::to_string(idxOfMyMachine) + "/all_gather_nccl_id"));
  }
  std::vector<NcclComm> reduceScatterComms = createManyNcclComms(
      idxOfMyFirstDevice, devices_, devicesPerMachine, reduceScatterUniqueId);
  std::vector<NcclComm> allGatherComms = createManyNcclComms(
      idxOfMyFirstDevice, devices_, devicesPerMachine, allGatherUniqueId);

  std::vector<NcclComm> collectComms;
  std::vector<NcclComm> diffuseComms;
  for (const auto deviceOffset : c10::irange(numDevices)) {
    ncclUniqueId collectUniqueId;
    ncclUniqueId diffuseUniqueId;
    if (idxOfMyMachine == 0) {
      NCCL_CHECK(ncclGetUniqueId(&collectUniqueId));
      NCCL_CHECK(ncclGetUniqueId(&diffuseUniqueId));
      store->set(
          "devices/" + std::to_string(idxOfMyFirstDevice + deviceOffset) +
              "/collect_nccl_id",
          podToByteString(collectUniqueId));
      store->set(
          "devices/" + std::to_string(idxOfMyFirstDevice + deviceOffset) +
              "/diffuse_nccl_id",
          podToByteString(diffuseUniqueId));
    } else {
      collectUniqueId = byteStringToPod<ncclUniqueId>(store->get(
          "devices/" + std::to_string(idxOfMyFirstDevice + deviceOffset) +
          "/collect_nccl_id"));
      diffuseUniqueId = byteStringToPod<ncclUniqueId>(store->get(
          "devices/" + std::to_string(idxOfMyFirstDevice + deviceOffset) +
          "/diffuse_nccl_id"));
    }
    collectComms.push_back(createOneNcclComm(
        idxOfMyMachine, devices_[deviceOffset], numMachines, collectUniqueId));
    diffuseComms.push_back(createOneNcclComm(
        idxOfMyMachine, devices_[deviceOffset], numMachines, diffuseUniqueId));
  }

  nodes_.reserve(numDevices);
  for (const auto deviceOffset : c10::irange(numDevices)) {
    nodes_.push_back(std::make_unique<DeviceFairring>(
        devices_[deviceOffset].index(),
        idxOfMyMachine,
        idxOfMyFirstDevice + deviceOffset,
        numMachines,
        devicesPerMachine,
        deviceGlobalRankIsFavorable,
        store,
        std::move(reduceScatterComms[deviceOffset]),
        std::move(collectComms[deviceOffset]),
        std::move(diffuseComms[deviceOffset]),
        std::move(allGatherComms[deviceOffset]),
        maxMemoryAllocatedInBytes,
        maxPaddingAllocatedInBytes,
        minParallelism));
  }

  streams_.reserve(numDevices);
  for (const auto deviceOffset : c10::irange(numDevices)) {
    streams_.emplace_back(deviceOffset);
    deviceToOffset_.emplace(devices_[deviceOffset], deviceOffset);
  }
}

MachineFairring::~MachineFairring() {
  // This is kinda stupid: when waiting on a Future's value this synchonizes the
  // current CUDA stream with the Future, and also records the Future's DataPtrs
  // on those current streams. This means that later on, when those DataPtrs are
  // deleted, the caching allocator will try to record a CUDA event on those
  // streams, but those streams could have been deleted by then! This doesn't
  // happen with "regular" PyTorch streams, because they come from a pool of
  // "leaked" streams, which are never deleted. It only happens with our own
  // "external" streams, because we actually try to clean up after ourselves.
  // The lazy solution is to instead just leak our streams as well.
  for (CudaStream& s : streams_) {
    s.leak();
  }
}

c10::intrusive_ptr<c10::ivalue::Future> MachineFairring::allReduce(
    c10d::ReduceOp opType,
    std::vector<at::Tensor> tensors) {
  TORCH_CHECK(tensors.size() == nodes_.size());
  std::vector<c10::optional<at::Tensor>> tensorForClient(
      nodes_.size(), c10::nullopt);
  for (const at::Tensor& t : tensors) {
    c10::Device device = t.device();
    auto iter = deviceToOffset_.find(device);
    TORCH_CHECK(iter != deviceToOffset_.end());
    size_t offset = iter->second;
    TORCH_CHECK(!tensorForClient[offset].has_value());
    tensorForClient[offset] = t;
  }

  for (const at::Tensor& t : tensors) {
    TORCH_CHECK(t.is_cuda());
    TORCH_CHECK(t.is_non_overlapping_and_dense());
  }

  const at::Tensor& firstTensor = tensors[0];
  for (const at::Tensor& t : tensors) {
    TORCH_CHECK(t.scalar_type() == firstTensor.scalar_type());
    TORCH_CHECK(t.sizes() == firstTensor.sizes());
    TORCH_CHECK(t.strides() == firstTensor.strides());
  }

  // FIXME Support more operation types
  TORCH_CHECK(opType == c10d::ReduceOp::SUM);

  c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
      c10::ListType::ofTensors());
  futures.reserve(nodes_.size());
  for (const auto idx : c10::irange(nodes_.size())) {
    const std::unique_ptr<DeviceFairring>& node = nodes_[idx];
    futures.push_back(node->allReduce(opType, tensorForClient[idx].value()));
  }

  if (futures.size() == 1) {
    return futures.extract(0);
  }

  auto multiFut = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::ofTensors(), devices_);

  std::vector<c10::cuda::CUDAStream> streams;
  for (const CudaStream& s : streams_) {
    streams.push_back(s);
  }

  c10::collectAll(futures)->addCallback([streams = std::move(streams),
                                         multiFut](c10::ivalue::Future& fut) {
    if (fut.hasError()) {
      multiFut->setError(fut.exception_ptr());
      return;
    }
    c10::cuda::CUDAMultiStreamGuard g(streams);
    std::vector<at::Tensor> tensors;
    tensors.reserve(streams.size());
    for (c10::IValue childVal : fut.value().toListRef()) {
      c10::intrusive_ptr<c10::ivalue::Future> childFut = childVal.toFuture();
      if (childFut->hasError()) {
        multiFut->setError(childFut->exception_ptr());
        return;
      }
      childFut->wait();
      tensors.push_back(childFut->value().toList().extract(0).toTensor());
    }
    multiFut->markCompleted(std::move(tensors));
  });

  return multiFut;
}

} // namespace fairring
