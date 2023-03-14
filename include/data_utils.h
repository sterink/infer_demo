#pragma once
#include "cuda_runtime.h"
#include "ixrt.h"

namespace rt = iluvatar::inferrt;

template <typename AllocFunc, typename FreeFunc>
class Buffer {
   public:
    using Ptr = std::shared_ptr<Buffer>;
    Buffer(uint64_t size, rt::TensorDataType data_type) : Buffer() { Resize(size_, data_type); }

    Buffer(const rt::TensorShape& shape) : Buffer() { Resize(shape); }

    ~Buffer() {
        if (buffer_) {
            freeFn(buffer_);
        }
    }

    Buffer() : size_(0), bytes_(0), buffer_(nullptr), data_type_(rt::TENSOR_DATA_TYPE_UNKNOWN) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer& operator=(Buffer&& buf) {
        if (this != &buf) {
            if (buffer_) {
                free(buffer_);
            }
            size_ = buf.size_;
            bytes_ = buf.bytes_;
            data_type_ = buf.data_type_;
            buffer_ = buf.buffer_;

            buf.buffer_ = nullptr;
            buf.size_ = 0;
            buf.bytes_ = 0;
        }
        return *this;
    }

    void Resize(uint64_t size, rt::TensorDataType data_type) {
        size_ = size;
        data_type_ = data_type;
        bytes_ = size * GetDataTypeBytes(data_type);

        if (not allocFn(&buffer_, bytes_)) {
            std::cerr << "Allocate memory for failed with bytes: " << bytes_ << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void Resize(const rt::TensorShape& shape) {
        uint64_t num_element = 1;
        for (auto i = 0; i < shape.dims.size(); ++i) {
            num_element *= shape.dims.at(i) + shape.padding.at(i);
        }
        Resize(num_element, shape.data_type);
    }

    void* GetDataPtr() { return buffer_; }

    uint64_t GetBytes() { return bytes_; }

    uint64_t GetSize() { return size_; }

    rt::TensorDataType GetDataType() { return data_type_; }

    uint64_t GetDataTypeBytes(rt::TensorDataType type) {
        switch (type) {
            case rt::TENSOR_DATA_TYPE_INT8:
                return 1;
            case rt::TENSOR_DATA_TYPE_FLOAT16:
                return 2;
            case rt::TENSOR_DATA_TYPE_FLOAT32:
            case rt::TENSOR_DATA_TYPE_INT32:
                return 4;
            case rt::TENSOR_DATA_TYPE_FLOAT64:
                return 8;
            default:
                return 0;
        }
    }

   private:
    void* buffer_;
    uint64_t size_;
    uint64_t bytes_;
    rt::TensorDataType data_type_;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator {
   public:
    bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

class DeviceFree {
   public:
    void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator {
   public:
    bool operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
   public:
    void operator()(void* ptr) const { free(ptr); }
};

class HostPinnedAllocator {
   public:
    bool operator()(void** ptr, size_t size) const { return cudaMallocHost(ptr, size) == cudaSuccess; }
};

class HostPinnedFree {
   public:
    void operator()(void* ptr) const { cudaFreeHost(ptr); }
};

using DeviceBuffer = Buffer<DeviceAllocator, DeviceFree>;
using HostBuffer = Buffer<HostAllocator, HostFree>;
using HostPinnedBuffer = Buffer<HostPinnedAllocator, HostPinnedFree>;

void SetRandomData(float* data, uint64_t size);
