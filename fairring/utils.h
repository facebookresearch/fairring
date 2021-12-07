/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <memory>
#include <stdexcept>

#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/tensorpipe_cuda.h>
#include <torch/torch.h>

#define DELETE_COPY_MOVE_CONSTRUCTORS(clsname) \
  clsname(const clsname&) = delete;            \
  clsname(clsname&&) = delete;                 \
  clsname& operator=(const clsname&) = delete; \
  clsname& operator=(clsname&&) = delete;

#define MY_CHECK(cond)                                  \
  if (!(cond)) {                                        \
    LOG(ERROR) << "Condition " << #cond << " is false"; \
    std::terminate();                                   \
  }

#define CUDA_CHECK(op)                                                    \
  {                                                                       \
    cudaError_t res = (op);                                               \
    if (res != cudaSuccess) {                                             \
      auto error_unused __attribute__((__unused__)) = cudaGetLastError(); \
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(res);            \
      throw std::runtime_error(                                           \
          std::string("CUDA error: ") + cudaGetErrorString(res));         \
    }                                                                     \
  }

#define NCCL_CHECK(op)                        \
  {                                           \
    ncclResult_t res = (op);                  \
    if (res != ncclSuccess) {                 \
      LOG(ERROR) << "NCCL error";             \
      throw std::runtime_error("NCCL error"); \
    }                                         \
  }

namespace fairring {

struct CudaStreamDeleter {
  void operator()(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
};

class CudaStream {
 public:
  CudaStream() {}

  CudaStream(c10::DeviceIndex index) : index_(index) {
    c10::cuda::CUDAGuard g(index);
    cudaStream_t stream;
    CUDA_CHECK(
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
    stream_ = decltype(stream_)(stream, CudaStreamDeleter{});
  }

  operator c10::Stream() const {
    return c10::cuda::getStreamFromExternal(stream_.get(), index_);
  }

  operator c10::cuda::CUDAStream() const {
    return c10::cuda::getStreamFromExternal(stream_.get(), index_);
  }

  operator cudaStream_t() const {
    return stream_.get();
  }

  void leak() {
    // This resets the CudaStream to its empty status, without really destroying
    // the CUDA stream, hence effectively leaking it. Do not use without a good
    // reason.
    stream_.release();
  }

 private:
  c10::DeviceIndex index_;
  std::unique_ptr<std::remove_pointer<cudaStream_t>::type, CudaStreamDeleter>
      stream_;
};

struct NcclCommDeleter {
  void operator()(ncclComm_t comm) {
    NCCL_CHECK(ncclCommDestroy(comm));
  }
};

using NcclComm =
    std::unique_ptr<std::remove_pointer<ncclComm_t>::type, NcclCommDeleter>;

inline std::vector<NcclComm> createManyNcclComms(
    int rankStart,
    const std::vector<c10::Device>& devices,
    int worldSize,
    ncclUniqueId uniqueId) {
  std::vector<ncclComm_t> rawComms(devices.size());
  NCCL_CHECK(ncclGroupStart());
  for (const auto deviceOffset : c10::irange(devices.size())) {
    c10::cuda::CUDAGuard g(devices[deviceOffset]);
    // std::ostringstream oss;
    // oss << "Initing NCCL on rank " << rankStart + deviceOffset << "/" <<
    // worldSize << " with unique ID "; for (size_t offset = 0; offset <
    // sizeof(ncclUniqueId); offset += 1) {
    //   oss << std::hex << std::setw(2) << std::setfill('0') <<
    //   static_cast<uint64_t>(*(reinterpret_cast<uint8_t*>(&uniqueId) +
    //   offset));
    // }
    // oss << std::endl;
    // std::cerr << oss.str();
    NCCL_CHECK(ncclCommInitRank(
        &rawComms[deviceOffset],
        worldSize,
        uniqueId,
        rankStart + deviceOffset));
  }
  NCCL_CHECK(ncclGroupEnd());

  std::vector<NcclComm> comms;
  comms.reserve(devices.size());
  for (const auto deviceOffset : c10::irange(devices.size())) {
    comms.push_back(NcclComm(rawComms[deviceOffset], NcclCommDeleter{}));
  }
  return comms;
}

inline std::vector<uint8_t> stringToByteVector(const std::string& s) {
  std::vector<uint8_t> v(s.size());
  std::memcpy(v.data(), s.data(), s.size());
  return v;
}

inline std::string byteVectorToString(const std::vector<uint8_t>& v) {
  std::string s(v.data(), v.data() + v.size());
  return s;
}

inline std::vector<uint8_t> integerToByteVector(int64_t n) {
  return stringToByteVector(std::to_string(n));
}

inline int64_t byteVectorToInteger(const std::vector<uint8_t>& v) {
  std::string s = byteVectorToString(v);
  return std::strtoll(
      reinterpret_cast<const char*>(s.c_str()),
      /*str_end=*/nullptr,
      /*base=*/10);
}

template <
    typename T,
    typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type =
        true>
inline std::vector<uint8_t> podToByteString(const T& t) {
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(&t),
      reinterpret_cast<const uint8_t*>(&t) + sizeof(T));
}

template <
    typename T,
    typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type =
        true>
inline T byteStringToPod(const std::vector<uint8_t>& v) {
  T t;
  std::memcpy(&t, v.data(), sizeof(T));
  return t;
}

inline std::shared_ptr<tensorpipe::Context> createTpContext(std::string name) {
  auto ctx = std::make_shared<tensorpipe::Context>(
      tensorpipe::ContextOptions().name(std::move(name)));
  ctx->registerTransport(0, "ibv", tensorpipe::transport::ibv::create());
  ctx->registerChannel(0, "cuda_gdr", tensorpipe::channel::cuda_gdr::create());
  ctx->registerChannel(1, "cuda_ipc", tensorpipe::channel::cuda_ipc::create());
  ctx->registerChannel(2, "cuda_xth", tensorpipe::channel::cuda_xth::create());
  return ctx;
}

inline ncclDataType_t torchToNcclDtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
      return ncclUint8;
    case c10::ScalarType::Char:
      return ncclChar;
    case c10::ScalarType::Int:
      return ncclInt;
    case c10::ScalarType::Long:
      return ncclInt64;
    case c10::ScalarType::Half:
      return ncclHalf;
    case c10::ScalarType::Float:
      return ncclFloat;
    case c10::ScalarType::Double:
      return ncclDouble;
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    case c10::ScalarType::BFloat16:
      return ncclBfloat16;
#endif
    default:
      MY_CHECK(false);
  }
}

inline at::Tensor viewAsDtype(
    at::Tensor byteTensor,
    c10::ScalarType dtype,
    size_t sizeInElems) {
  MY_CHECK(byteTensor.scalar_type() == c10::kByte);
  return torch::from_blob(
      byteTensor.data_ptr(),
      {static_cast<long>(sizeInElems)},
      byteTensor.options().dtype(dtype));
}

template <typename T>
inline constexpr T ceilOfRatio(T num, T den) {
  if (num == 0) {
    return 0;
  }
  return (num - 1) / den + 1;
}

template <typename T>
inline constexpr T roundDownToNearestMultiple(T val, T factor) {
  return (val / factor) * factor;
}

template <typename T>
inline constexpr T roundUpToNearestMultiple(T val, T factor) {
  return ceilOfRatio(val, factor) * factor;
}

// Each shard/slot/slice must contain a whole number of elements, whether they
// are halfs, floats or doubles. We thus align them to (a multiple of) this
// number of bytes, which is the size of doubles, the largest data type.
static constexpr size_t kAlignment = 8;

// The drivers of some InfiniBand cards appear to SEGFAULT when handling with
// messages that are too small. (Our messages, over TensorPipe, are the shards).
// Hence we enforce a minimum shard size, at the cost of sending garbage data
// inside some of the shards.
static constexpr size_t kMinShardSizeInBytes = 64;
static_assert(kMinShardSizeInBytes % kAlignment == 0, "");

struct Layout {
  size_t shardSizeInBytes;
  size_t slotSizeInBytes;
  size_t sliceSizeInBytes;
  size_t numSlots;
};

inline Layout computeLayout(
    size_t maxMemoryAllocatedInBytes,
    size_t sliceSizeInBytes,
    size_t numMachines,
    size_t numDevicesPerMachine) {
  // Constraints:
  // sliceSize == slotSize * numDevicesPerMachine
  // slotSize <= shardSize * numMachines
  // shardSize % kAlignment == 0

  // Memory usage:
  // - The server uses:
  //   * numMachines * shardSize bytes into which it stages incoming requests
  //   * shardSize bytes into which it aggregates the stages data before sending
  // - The client uses:
  //   * slotSize into which it puts the result of reduceScatter before sending,
  //     and the incoming responses before calling allGather
  //   * numDevicesPerMachine * kAlignment as temporary storage used to pad the
  //     excess data that cannot be handled by reduceScatter
  // All of the above are repeated for each slot.

  // We could "fix" this by tweaking the sliceSize, but it's probably not good:
  // - if we decrease it, and the user passes in tensors of that size, we'd end
  //   up slicing each tensor into two (with the second half being tiny) while
  //   the user believe they have a perfect 1:1 correspondence
  // - if we increase it, and the user passes in tensors of that size, it means
  //   we need to add some padding to each tensor we handle, which is a more
  //   inefficient path.
  MY_CHECK(sliceSizeInBytes % (kAlignment * numDevicesPerMachine) == 0);

  size_t slotSizeInBytes = sliceSizeInBytes / numDevicesPerMachine;
  size_t shardSizeInBytes = roundUpToNearestMultiple(
      ceilOfRatio(slotSizeInBytes, numMachines), kAlignment);
  size_t totalMemoryNeededPerSlot =
      ((2 * numMachines + 1) * shardSizeInBytes +
       numDevicesPerMachine * kAlignment);
  size_t numSlots = maxMemoryAllocatedInBytes / totalMemoryNeededPerSlot;

  MY_CHECK(shardSizeInBytes >= kMinShardSizeInBytes);
  MY_CHECK(slotSizeInBytes >= kMinShardSizeInBytes * numMachines);

  LOG(WARNING) << "The Fairring process group will achieve a parallelism of "
               << numSlots;

  return Layout{
      .shardSizeInBytes = shardSizeInBytes,
      .slotSizeInBytes = slotSizeInBytes,
      .sliceSizeInBytes = sliceSizeInBytes,
      .numSlots = numSlots,
  };
}

// Could be constexpr, if it weren't for
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102878
inline size_t getShardOffsetInBytes(
    size_t machineIdx,
    size_t numMachines,
    size_t slotSizeInElems,
    size_t elementSizeInBytes) {
  MY_CHECK(kAlignment % elementSizeInBytes == 0);
  return std::max(
      machineIdx * kMinShardSizeInBytes,
      machineIdx * slotSizeInElems / numMachines * elementSizeInBytes);
}

// Could be constexpr, if it weren't for
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102878
inline size_t getShardSizeInBytes(
    size_t machineIdx,
    size_t numMachines,
    size_t slotSizeInElems,
    size_t elementSizeInBytes) {
  return getShardOffsetInBytes(
             machineIdx + 1, numMachines, slotSizeInElems, elementSizeInBytes) -
      getShardOffsetInBytes(
             machineIdx, numMachines, slotSizeInElems, elementSizeInBytes);
}

class CommandQueue {
 public:
  CommandQueue() = default;

  void enqueue(std::function<void()> fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    fns_.push_back(std::move(fn));
    cv_.notify_all();
  }

  std::function<void()> dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !fns_.empty(); });
    std::function<void()> fn = std::move(fns_.front());
    fns_.pop_front();
    return fn;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> fns_;
};

} // namespace fairring
