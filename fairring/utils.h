/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <limits>
#include <memory>
#include <stdexcept>

#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
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

#define CUDA_CHECK(op)                                                      \
  {                                                                         \
    cudaError_t res = (op);                                                 \
    if (res != cudaSuccess) {                                               \
      auto error_unused __attribute__((__unused__)) = cudaGetLastError();   \
      LOG(ERROR) << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(res);                                \
      throw std::runtime_error(                                             \
          std::string("CUDA error: ") + cudaGetErrorString(res));           \
    }                                                                       \
  }

inline std::string getNcclErrorString(ncclResult_t err) {
  switch (err) {
    case ncclSuccess:
      return "success";
    case ncclUnhandledCudaError:
      return "unhandled CUDA error";
    case ncclSystemError:
      return "system error";
    case ncclInternalError:
      return "internal error";
    case ncclInvalidArgument:
      return "invalid argument";
    case ncclInvalidUsage:
      return "invalid usage";
    case ncclNumResults:
      break;
  }
  return "unknown error type";
}

#define NCCL_CHECK(op)                                                      \
  {                                                                         \
    ncclResult_t res = (op);                                                \
    if (res != ncclSuccess) {                                               \
      LOG(ERROR) << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << getNcclErrorString(res);                                \
      throw std::runtime_error(                                             \
          std::string("NCCL error: ") + getNcclErrorString(res));           \
    }                                                                       \
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

  explicit CudaStream(c10::DeviceIndex index) : index_(index) {
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

inline NcclComm createOneNcclComm(
    int rank,
    c10::Device device,
    int worldSize,
    ncclUniqueId uniqueId) {
  ncclComm_t rawComm;
  c10::cuda::CUDAGuard g(device);
  NCCL_CHECK(ncclCommInitRank(&rawComm, worldSize, uniqueId, rank));
  return NcclComm(rawComm, NcclCommDeleter{});
}

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
    // worldSize << " with unique ID "; for (int64_t offset = 0; offset <
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

inline ncclRedOp_t torchToNcclRedOp(c10d::ReduceOp opType) {
  switch (opType) {
    case c10d::ReduceOp::SUM:
      return ncclSum;
    case c10d::ReduceOp::MAX:
      return ncclMax;
    default:
      MY_CHECK(false);
  }
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
constexpr int64_t kAlignment = 8;

struct Layout {
  int64_t slotSizeInBytes;
  int64_t sliceSizeInBytes;
  int64_t numPaddingSlots;
  int64_t numStagingSlots;
};

inline Layout computeLayout(
    int64_t maxMemoryAllocatedInBytes,
    int64_t maxPaddingAllocatedInBytes,
    int64_t minParallelism,
    int64_t numMachines,
    int64_t numDevicesPerMachine) {
  int64_t onePaddingSizeInBytes =
      numDevicesPerMachine * numMachines * kAlignment;
  int64_t numPaddingSlots = maxPaddingAllocatedInBytes / onePaddingSizeInBytes;
  MY_CHECK(minParallelism <= numPaddingSlots);

  int64_t sliceSizeInBytes = std::numeric_limits<int64_t>::max();
  int64_t slotSizeInBytes = 0;
  int64_t numStagingSlots = 0;
  if (numDevicesPerMachine == 1) {
    int64_t paddingMemoryInBytes =
        2 * minParallelism * numDevicesPerMachine * numMachines * kAlignment;
    MY_CHECK(paddingMemoryInBytes <= maxMemoryAllocatedInBytes);
    slotSizeInBytes = roundDownToNearestMultiple(
        (maxMemoryAllocatedInBytes - paddingMemoryInBytes) / minParallelism,
        numDevicesPerMachine * numMachines * kAlignment);
    sliceSizeInBytes = slotSizeInBytes * numDevicesPerMachine;
    numStagingSlots = minParallelism;
    numPaddingSlots = minParallelism;
  }

  LOG(WARNING) << "The Fairring process group will achieve a parallelism of "
               << numPaddingSlots << (numStagingSlots == 0 ? "+" : "")
               << " and its slice size is "
               << (sliceSizeInBytes == std::numeric_limits<int64_t>::max()
                       ? "infinite"
                       : std::to_string(sliceSizeInBytes));

  return Layout{
      .slotSizeInBytes = slotSizeInBytes,
      .sliceSizeInBytes = sliceSizeInBytes,
      .numPaddingSlots = numPaddingSlots,
      .numStagingSlots = numStagingSlots,
  };
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

inline at::Tensor viewAsFlat(const at::Tensor& t) {
  MY_CHECK(t.layout() == at::kStrided);
  MY_CHECK(t.is_non_overlapping_and_dense());
  auto flatTImpl = c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(t.storage()), t.key_set(), t.dtype());
  flatTImpl->set_storage_offset(t.storage_offset());
  flatTImpl->set_sizes_contiguous({t.numel()});
  return at::Tensor(std::move(flatTImpl));
}

inline at::Tensor unUnbind(const std::vector<at::Tensor>& ts) {
  MY_CHECK(!ts.empty());
  for (const auto idx : c10::irange(ts.size())) {
    const at::Tensor& t = ts[idx];
    MY_CHECK(t.layout() == at::kStrided);
    MY_CHECK(t.is_non_overlapping_and_dense());
    MY_CHECK(t.numel() == ts[0].numel());
    MY_CHECK(t.storage() == ts[0].storage());
    MY_CHECK(
        t.storage_offset() ==
        ts[0].storage_offset() + static_cast<int64_t>(idx) * ts[0].numel());
    MY_CHECK(t.key_set() == ts[0].key_set());
    MY_CHECK(t.dtype() == ts[0].dtype());
  }
  auto catTImpl = c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(ts[0].storage()),
      ts[0].key_set(),
      ts[0].dtype());
  catTImpl->set_storage_offset(ts[0].storage_offset());
  catTImpl->set_sizes_contiguous(
      {ts[0].numel() * static_cast<int64_t>(ts.size())});
  return at::Tensor(std::move(catTImpl));
}

} // namespace fairring
