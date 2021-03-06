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

std::string getBootID() {
  std::ifstream f{"/proc/sys/kernel/random/boot_id"};
  MY_CHECK(f.is_open());
  std::string v;
  getline(f, v);
  f.close();
  return v;
}

} // namespace

namespace fairring {

DeploymentInfo detectDeploymentInfo(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    int64_t numDevices) {
  std::string machineId = getBootID();
  TORCH_CHECK(numDevices > 0);
  store->set(
      "rdv/" + std::to_string(rank) + "/machine_idx",
      stringToByteVector(machineId));
  store->set(
      "rdv/" + std::to_string(rank) + "/num_devices",
      integerToByteVector(numDevices));

  struct Machine {
    int64_t idx = 0;
    int64_t totalDevices = 0;
  };
  std::unordered_map<std::string, Machine> idToMachine;
  struct Device {
    int64_t machineIdx = 0;
    int64_t idxWithinMachine = 0;
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
      int64_t newMachineIdx = idToMachine.size();
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
  int64_t numMachines = idToMachine.size();
  int64_t devicesPerMachine = idToMachine.begin()->second.totalDevices;
  for (const auto& kv : idToMachine) {
    TORCH_CHECK(kv.second.totalDevices == devicesPerMachine);
  }

  at::Tensor deviceGlobalRankAssignment =
      at::empty({numMachines, devicesPerMachine}, at::kLong);
  for (const auto deviceGlobalRank :
       c10::irange(numMachines * devicesPerMachine)) {
    const Device& device = allDevices[deviceGlobalRank];
    deviceGlobalRankAssignment[device.machineIdx][device.idxWithinMachine] =
        deviceGlobalRank;
  }
  bool deviceGlobalRankIsFavorable;
  if (deviceGlobalRankAssignment.equal(
          torch::arange(numMachines * devicesPerMachine)
              .view({devicesPerMachine, numMachines})
              .transpose(0, 1))) {
    deviceGlobalRankIsFavorable = true;
  } else if (deviceGlobalRankAssignment.equal(
                 torch::arange(numMachines * devicesPerMachine)
                     .view({numMachines, devicesPerMachine}))) {
    deviceGlobalRankIsFavorable = false;
  } else {
    MY_CHECK(false);
  }

  return DeploymentInfo{
      .numDevices = numDevices,
      .numMachines = numMachines,
      .idxOfMyFirstDevice = idxOfMyFirstDevice,
      .idxOfMyMachine = idxOfMyMachine,
      .devicesPerMachine = devicesPerMachine,
      .deviceGlobalRankIsFavorable = deviceGlobalRankIsFavorable};
}

MachineFairring::MachineFairring(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int size,
    std::vector<c10::Device> devices,
    int64_t maxMemoryAllocatedInBytes,
    int64_t maxPaddingAllocatedInBytes,
    int64_t minParallelism)
    : store_(std::move(store)), devices_(std::move(devices)) {
  TORCH_CHECK(0 <= rank && rank < size);
  deploymentInfo_ = detectDeploymentInfo(store_, rank, size, devices_.size());

  nodes_.reserve(deploymentInfo_.numDevices);
  for (const auto deviceOffset : c10::irange(deploymentInfo_.numDevices)) {
    nodes_.push_back(std::make_unique<DeviceFairring>(
        devices_[deviceOffset].index(),
        deploymentInfo_.idxOfMyMachine,
        deploymentInfo_.idxOfMyFirstDevice + deviceOffset,
        deploymentInfo_.numMachines,
        deploymentInfo_.devicesPerMachine,
        deploymentInfo_.deviceGlobalRankIsFavorable,
        store,
        maxMemoryAllocatedInBytes,
        maxPaddingAllocatedInBytes,
        minParallelism));
  }

  streams_.reserve(deploymentInfo_.numDevices);
  for (const auto deviceOffset : c10::irange(deploymentInfo_.numDevices)) {
    streams_.emplace_back(devices_[deviceOffset].index());
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
  // FIXME Support more operation types
  MY_CHECK(opType == c10d::ReduceOp::SUM || opType == c10d::ReduceOp::MAX);

  MY_CHECK(tensors.size() == devices_.size());
  for (const auto deviceOffset : c10::irange(tensors.size())) {
    MY_CHECK(tensors[deviceOffset].device() == devices_[deviceOffset]);
  }

  ensureReduceScatterCommsEstablished();
  ensureCollectCommsEstablished();
  ensureDiffuseCommsEstablished();
  ensureAllGatherCommsEstablished();

  c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
      c10::ListType::ofTensors());
  futures.reserve(nodes_.size());
  for (const auto idx : c10::irange(nodes_.size())) {
    const std::unique_ptr<DeviceFairring>& node = nodes_[idx];
    futures.push_back(node->allReduce(opType, tensors[idx]));
  }

  return mergeMultiDeviceFutures(std::move(futures));
}

c10::intrusive_ptr<c10::ivalue::Future> MachineFairring::reduceScatter(
    c10d::ReduceOp opType,
    std::vector<TensorPair> tensors) {
  // FIXME Support more operation types
  MY_CHECK(opType == c10d::ReduceOp::SUM || opType == c10d::ReduceOp::MAX);

  MY_CHECK(tensors.size() == devices_.size());
  for (const auto deviceOffset : c10::irange(tensors.size())) {
    MY_CHECK(tensors[deviceOffset].output.device() == devices_[deviceOffset]);
  }

  ensureReduceScatterCommsEstablished();
  ensureCollectCommsEstablished();

  c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
      c10::ListType::ofTensors());
  futures.reserve(nodes_.size());
  for (const auto idx : c10::irange(nodes_.size())) {
    const std::unique_ptr<DeviceFairring>& node = nodes_[idx];
    futures.push_back(
        node->reduceScatter(opType, tensors[idx].input, tensors[idx].output));
  }

  return mergeMultiDeviceFutures(std::move(futures));
}

c10::intrusive_ptr<c10::ivalue::Future> MachineFairring::allGather(
    std::vector<TensorPair> tensors) {
  MY_CHECK(tensors.size() == devices_.size());
  for (const auto deviceOffset : c10::irange(tensors.size())) {
    MY_CHECK(tensors[deviceOffset].input.device() == devices_[deviceOffset]);
  }

  ensureDiffuseCommsEstablished();
  ensureAllGatherCommsEstablished();

  c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
      c10::ListType::ofTensors());
  futures.reserve(nodes_.size());
  for (const auto idx : c10::irange(nodes_.size())) {
    const std::unique_ptr<DeviceFairring>& node = nodes_[idx];
    futures.push_back(node->allGather(tensors[idx].input, tensors[idx].output));
  }

  return mergeMultiDeviceFutures(std::move(futures));
}

c10::intrusive_ptr<c10::ivalue::Future> MachineFairring::
    mergeMultiDeviceFutures(
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures) {
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

std::vector<NcclComm> MachineFairring::establishReduceScatterComms() {
  ncclUniqueId reduceScatterUniqueId;
  if (deploymentInfo_.idxOfMyFirstDevice == 0) {
    NCCL_CHECK(ncclGetUniqueId(&reduceScatterUniqueId));
    store_->set(
        "machines/" + std::to_string(deploymentInfo_.idxOfMyMachine) +
            "/reduce_scatter_nccl_id",
        podToByteString(reduceScatterUniqueId));
  } else {
    reduceScatterUniqueId = byteStringToPod<ncclUniqueId>(store_->get(
        "machines/" + std::to_string(deploymentInfo_.idxOfMyMachine) +
        "/reduce_scatter_nccl_id"));
  }
  return createManyNcclComms(
      deploymentInfo_.idxOfMyFirstDevice,
      devices_,
      deploymentInfo_.devicesPerMachine,
      reduceScatterUniqueId);
}

std::vector<NcclComm> MachineFairring::establishCollectComms() {
  std::vector<NcclComm> collectComms;
  for (const auto deviceOffset : c10::irange(deploymentInfo_.numDevices)) {
    ncclUniqueId collectUniqueId;
    if (deploymentInfo_.idxOfMyMachine == 0) {
      NCCL_CHECK(ncclGetUniqueId(&collectUniqueId));
      store_->set(
          "devices/" +
              std::to_string(
                  deploymentInfo_.idxOfMyFirstDevice + deviceOffset) +
              "/collect_nccl_id",
          podToByteString(collectUniqueId));
    } else {
      collectUniqueId = byteStringToPod<ncclUniqueId>(store_->get(
          "devices/" +
          std::to_string(deploymentInfo_.idxOfMyFirstDevice + deviceOffset) +
          "/collect_nccl_id"));
    }
    collectComms.push_back(createOneNcclComm(
        deploymentInfo_.idxOfMyMachine,
        devices_[deviceOffset],
        deploymentInfo_.numMachines,
        collectUniqueId));
  }
  return collectComms;
}

std::vector<NcclComm> MachineFairring::establishDiffuseComms() {
  std::vector<NcclComm> diffuseComms;
  for (const auto deviceOffset : c10::irange(deploymentInfo_.numDevices)) {
    ncclUniqueId diffuseUniqueId;
    if (deploymentInfo_.idxOfMyMachine == 0) {
      NCCL_CHECK(ncclGetUniqueId(&diffuseUniqueId));
      store_->set(
          "devices/" +
              std::to_string(
                  deploymentInfo_.idxOfMyFirstDevice + deviceOffset) +
              "/diffuse_nccl_id",
          podToByteString(diffuseUniqueId));
    } else {
      diffuseUniqueId = byteStringToPod<ncclUniqueId>(store_->get(
          "devices/" +
          std::to_string(deploymentInfo_.idxOfMyFirstDevice + deviceOffset) +
          "/diffuse_nccl_id"));
    }
    diffuseComms.push_back(createOneNcclComm(
        deploymentInfo_.idxOfMyMachine,
        devices_[deviceOffset],
        deploymentInfo_.numMachines,
        diffuseUniqueId));
  }
  return diffuseComms;
}

std::vector<NcclComm> MachineFairring::establishAllGatherComms() {
  ncclUniqueId allGatherUniqueId;
  if (deploymentInfo_.idxOfMyFirstDevice == 0) {
    NCCL_CHECK(ncclGetUniqueId(&allGatherUniqueId));
    store_->set(
        "machines/" + std::to_string(deploymentInfo_.idxOfMyMachine) +
            "/all_gather_nccl_id",
        podToByteString(allGatherUniqueId));
  } else {
    allGatherUniqueId = byteStringToPod<ncclUniqueId>(store_->get(
        "machines/" + std::to_string(deploymentInfo_.idxOfMyMachine) +
        "/all_gather_nccl_id"));
  }
  return createManyNcclComms(
      deploymentInfo_.idxOfMyFirstDevice,
      devices_,
      deploymentInfo_.devicesPerMachine,
      allGatherUniqueId);
}

void MachineFairring::ensureReduceScatterCommsEstablished() {
  if (!establishedReduceScatterComm_) {
    std::vector<NcclComm> comms = establishReduceScatterComms();
    for (const auto idx : c10::irange(nodes_.size())) {
      nodes_[idx]->setReduceScatterComm(std::move(comms[idx]));
    }
    establishedReduceScatterComm_ = true;
  }
}

void MachineFairring::ensureCollectCommsEstablished() {
  if (!establishedCollectComm_) {
    std::vector<NcclComm> comms = establishCollectComms();
    for (const auto idx : c10::irange(nodes_.size())) {
      nodes_[idx]->setCollectComm(std::move(comms[idx]));
    }
    establishedCollectComm_ = true;
  }
}

void MachineFairring::ensureDiffuseCommsEstablished() {
  if (!establishedDiffuseComm_) {
    std::vector<NcclComm> comms = establishDiffuseComms();
    for (const auto idx : c10::irange(nodes_.size())) {
      nodes_[idx]->setDiffuseComm(std::move(comms[idx]));
    }
    establishedDiffuseComm_ = true;
  }
}

void MachineFairring::ensureAllGatherCommsEstablished() {
  if (!establishedAllGatherComm_) {
    std::vector<NcclComm> comms = establishAllGatherComms();
    for (const auto idx : c10::irange(nodes_.size())) {
      nodes_[idx]->setAllGatherComm(std::move(comms[idx]));
    }
    establishedAllGatherComm_ = true;
  }
}

} // namespace fairring
