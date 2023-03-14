#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace iluvatar::inferrt {

using ShapeTable = std::vector<std::pair<std::string, std::vector<int32_t>>>;
// TODO: Delete pipeline type from here, according to called method execute or
// enqueue
enum PipelineType { PIPELINE_TYPE_SYNCHRONOUS = 0, PIPELINE_TYPE_ASYNCHRONOUS };

enum TensorDataType {
    TENSOR_DATA_TYPE_UNKNOWN = 0,
    TENSOR_DATA_TYPE_INT8,
    TENSOR_DATA_TYPE_FLOAT16,
    TENSOR_DATA_TYPE_FLOAT32,
    TENSOR_DATA_TYPE_INT32,
    TENSOR_DATA_TYPE_INT64,
    TENSOR_DATA_TYPE_FLOAT64,
    TENSOR_DATA_TYPE_INT8_FIX,
    TENSOR_DATA_TYPE_FLOAT16_FIX,
    TENSOR_DATA_TYPE_FLOAT32_FIX,
    TENSOR_DATA_TYPE_INT32_FIX,
    TENSOR_DATA_TYPE_INT64_FIX,
    TENSOR_DATA_TYPE_FLOAT64_FIX,
};

using TypeTable = std::unordered_map<std::string, TensorDataType>;

enum TensorDataFormat {
    TENSOR_DATA_FORMAT_UNKNOWN = 0,  // As default layout, 1D sequence array
    TENSOR_DATA_FORMAT_LINEAR,
    TENSOR_DATA_FORMAT_NL,
    TENSOR_DATA_FORMAT_NLH,
    TENSOR_DATA_FORMAT_NCHW,
    TENSOR_DATA_FORMAT_NHWC,
    TENSOR_DATA_FORMAT_NCDHW,
    TENSOR_DATA_FORMAT_NDHWC,
};

enum TargetDevice { TARGET_DEVICE_UNKNOWN = 0, TARGET_DEVICE_CPU, TARGET_DEVICE_GPU };

enum RetCode { RET_CODE_SUCCESS = 0, RET_CODE_FAILURE };

struct RuntimeContext {
    TensorDataType data_type;
    TensorDataFormat data_format;
    TargetDevice device_type;
    PipelineType pipeline_type;
    TypeTable input_types;
    TypeTable output_types;
    TargetDevice input_device;
    TargetDevice output_device;
    RuntimeContext(TensorDataType in_type, TensorDataFormat in_format, TargetDevice in_device,
                   PipelineType in_pipeline = PIPELINE_TYPE_SYNCHRONOUS, const TypeTable& it = {},
                   const TypeTable& ot = {}, TargetDevice i_device = TARGET_DEVICE_CPU,
                   TargetDevice o_device = TARGET_DEVICE_CPU)
        : data_type(in_type),
          data_format(in_format),
          device_type(in_device),
          pipeline_type(in_pipeline),
          input_types(it),
          output_types(ot),
          input_device(i_device),
          output_device(o_device) {}
    RuntimeContext()
        : RuntimeContext(TENSOR_DATA_TYPE_FLOAT32, TENSOR_DATA_FORMAT_UNKNOWN, TARGET_DEVICE_GPU,
                         PIPELINE_TYPE_SYNCHRONOUS, {}, {}){};

    RuntimeContext(const RuntimeContext&) = default;
    RuntimeContext& operator=(const RuntimeContext&) = default;
};

struct RuntimeConfig {
    std::string graph_file;
    std::string quant_file;
    std::string weights_file;
    std::string engine_file;
    ShapeTable input_shapes;
    RuntimeContext runtime_context;
    int32_t device_idx;
    RuntimeConfig() = default;
    RuntimeConfig(const RuntimeConfig&) = default;
    RuntimeConfig& operator=(const RuntimeConfig&) = default;
};

struct TensorShape {
    std::vector<int32_t> dims;
    std::vector<int32_t> padding;
    TensorDataType data_type;
    TensorDataFormat data_format;
    TensorShape(const std::vector<int32_t>& in_dims, const std::vector<int32_t>& in_pad, TensorDataType in_dtype,
                TensorDataFormat in_dformat)
        : dims(in_dims), padding(in_pad), data_type(in_dtype), data_format(in_dformat) {}
    TensorShape() : data_type(TENSOR_DATA_TYPE_UNKNOWN), data_format(TENSOR_DATA_FORMAT_UNKNOWN) {}
    TensorShape(const TensorShape&) = default;
    TensorShape& operator=(const TensorShape&) = default;
};

struct IOBuffer {
    using Ptr = std::shared_ptr<IOBuffer>;
    std::string name;
    void* data;
    TensorShape shape;
    IOBuffer() : data(nullptr) {}
    IOBuffer(const std::string& n, void* d, const TensorShape& s = TensorShape()) : name(n), data(d), shape(s) {}
};

enum AlgoSelectMode { ALGO_SELECT_MODE_MANUAL = 0, ALGO_SELECT_MODE_AUTO_TIMEING };

using TensorShapeMap = std::unordered_map<std::string, TensorShape>;
using IOBuffers = std::vector<IOBuffer>;
using TaskMethod = std::function<void(void)>;
using PostProcessMethod = std::function<void(IOBuffers*, void*, int32_t)>;

}  // end of namespace iluvatar::inferrt
