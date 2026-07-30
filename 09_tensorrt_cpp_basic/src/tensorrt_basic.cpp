#include "tensorrt_basic.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lesson09 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

template <typename T>
struct TensorRtDeleter {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

template <typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const {
        return stream_;
    }

    void synchronize() const {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault),
                   "cudaEventCreateWithFlags");
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t byte_count) : byte_count_(byte_count) {
        if (byte_count_ == 0) {
            throw std::runtime_error("DeviceBuffer cannot allocate zero bytes.");
        }
        check_cuda(cudaMalloc(&ptr_, byte_count_), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        *this = std::move(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                (void)cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            byte_count_ = other.byte_count_;
            other.ptr_ = nullptr;
            other.byte_count_ = 0;
        }
        return *this;
    }

    void* get() const {
        return ptr_;
    }

    std::size_t byte_count() const {
        return byte_count_;
    }

private:
    void* ptr_ = nullptr;
    std::size_t byte_count_ = 0;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t byte_count) : byte_count_(byte_count) {
        if (byte_count_ == 0) {
            throw std::runtime_error("PinnedHostBuffer cannot allocate zero bytes.");
        }
        check_cuda(cudaMallocHost(&ptr_, byte_count_), "cudaMallocHost");
        std::fill_n(data(), byte_count_, std::uint8_t{0});
    }

    ~PinnedHostBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFreeHost(ptr_);
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept {
        *this = std::move(other);
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                (void)cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            byte_count_ = other.byte_count_;
            other.ptr_ = nullptr;
            other.byte_count_ = 0;
        }
        return *this;
    }

    void* get() const {
        return ptr_;
    }

    std::uint8_t* data() const {
        return static_cast<std::uint8_t*>(ptr_);
    }

    std::size_t byte_count() const {
        return byte_count_;
    }

private:
    void* ptr_ = nullptr;
    std::size_t byte_count_ = 0;
};

struct TensorBuffer {
    TensorSummary summary;
    std::unique_ptr<DeviceBuffer> device;
    PinnedHostBuffer host;

    explicit TensorBuffer(TensorSummary tensor_summary)
        : summary(std::move(tensor_summary)), host(summary.byte_count) {}

    void* tensor_address() const {
        return device ? device->get() : host.get();
    }
};

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    const std::ifstream::pos_type end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error("File is empty: " + path);
    }

    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return bytes;
}

bool file_exists(const std::string& path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 && S_ISREG(status.st_mode);
}

std::vector<char> read_optional_binary_file(const std::string& path) {
    if (path.empty() || !file_exists(path)) {
        return {};
    }
    return read_binary_file(path);
}

std::string default_timing_cache_path(bool allow_tf32) {
    return allow_tf32 ? "outputs/tensorrt_timing_tf32.cache"
                      : "outputs/tensorrt_timing_fp32.cache";
}

std::string resolve_timing_cache_path(const AppConfig& config) {
    if (!config.timing_cache_path.empty()) {
        return config.timing_cache_path;
    }
    return default_timing_cache_path(config.allow_tf32);
}

void make_directory(const std::string& path) {
    if (path.empty()) {
        return;
    }
    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return;
    }
    throw std::runtime_error("Failed to create directory " + path + ": " + std::strerror(errno));
}

void ensure_parent_directories(const std::string& path) {
    std::size_t slash = path.find('/');
    while (slash != std::string::npos) {
        const std::string directory = path.substr(0, slash);
        if (!directory.empty() && directory != "." && directory != "..") {
            make_directory(directory);
        }
        slash = path.find('/', slash + 1);
    }
}

void write_binary_file(const std::string& path, const void* data, std::size_t byte_count) {
    ensure_parent_directories(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(byte_count));
    if (!output) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}

std::size_t workspace_bytes(std::size_t workspace_mib) {
    constexpr std::size_t mib = 1024U * 1024U;
    if (workspace_mib > std::numeric_limits<std::size_t>::max() / mib) {
        throw std::runtime_error("workspace_mib is too large.");
    }
    return workspace_mib * mib;
}

std::string to_string(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        case nvinfer1::DataType::kUINT8:
            return "uint8";
        case nvinfer1::DataType::kFP8:
            return "fp8";
        case nvinfer1::DataType::kBF16:
            return "bfloat16";
        case nvinfer1::DataType::kINT64:
            return "int64";
        case nvinfer1::DataType::kINT4:
            return "int4";
        case nvinfer1::DataType::kFP4:
            return "fp4";
        case nvinfer1::DataType::kE8M0:
            return "e8m0";
    }
    return "unknown";
}

std::string to_string(nvinfer1::TensorIOMode mode) {
    switch (mode) {
        case nvinfer1::TensorIOMode::kINPUT:
            return "input";
        case nvinfer1::TensorIOMode::kOUTPUT:
            return "output";
        case nvinfer1::TensorIOMode::kNONE:
            return "none";
    }
    return "unknown";
}

std::string to_string(nvinfer1::TensorLocation location) {
    switch (location) {
        case nvinfer1::TensorLocation::kDEVICE:
            return "device";
        case nvinfer1::TensorLocation::kHOST:
            return "host";
    }
    return "unknown";
}

std::size_t bit_size(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 32;
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kBF16:
            return 16;
        case nvinfer1::DataType::kINT64:
            return 64;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kFP8:
        case nvinfer1::DataType::kE8M0:
            return 8;
        case nvinfer1::DataType::kINT4:
        case nvinfer1::DataType::kFP4:
            return 4;
    }
    throw std::runtime_error("Unsupported TensorRT data type.");
}

nvinfer1::Dims to_dims(const std::vector<int32_t>& dimensions) {
    if (dimensions.empty() ||
        dimensions.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        throw std::runtime_error("Input shape rank is outside TensorRT Dims limits.");
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(dimensions.size());
    std::copy(dimensions.begin(), dimensions.end(), dims.d);
    return dims;
}

std::vector<int64_t> to_vector(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        throw std::runtime_error("TensorRT returned an invalid shape.");
    }

    std::vector<int64_t> values;
    values.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        values.push_back(dims.d[i]);
    }
    return values;
}

bool has_dynamic_dimension(const nvinfer1::Dims& dims) {
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int64_t value) { return value < 0; });
}

const InputShape* find_input_shape(const std::vector<InputShape>& input_shapes,
                                   const std::string& tensor_name) {
    const auto found = std::find_if(input_shapes.begin(), input_shapes.end(),
                                    [&](const InputShape& shape) {
                                        return shape.tensor_name == tensor_name;
                                    });
    return found == input_shapes.end() ? nullptr : &(*found);
}

std::size_t checked_volume(const nvinfer1::Dims& dims, const std::string& tensor_name) {
    if (dims.nbDims <= 0) {
        throw std::runtime_error("Tensor " + tensor_name + " has invalid rank.");
    }

    std::size_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Tensor " + tensor_name +
                                     " still has a dynamic or invalid dimension.");
        }
        const auto dim = static_cast<std::size_t>(dims.d[i]);
        if (volume > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor " + tensor_name + " has too many elements.");
        }
        volume *= dim;
    }
    return volume;
}

std::size_t checked_byte_count(const nvinfer1::ICudaEngine& engine,
                               const std::string& tensor_name,
                               const nvinfer1::Dims& dims) {
    const std::size_t volume = checked_volume(dims, tensor_name);
    if (engine.getTensorVectorizedDim(tensor_name.c_str()) != -1) {
        throw std::runtime_error("Tensor " + tensor_name +
                                 " uses a vectorized IO format that this lesson does not support.");
    }
    const std::size_t bits = bit_size(engine.getTensorDataType(tensor_name.c_str()));
    if (volume > (std::numeric_limits<std::size_t>::max() - 7U) / bits) {
        throw std::runtime_error("Tensor " + tensor_name + " byte count overflowed.");
    }
    return (volume * bits + 7U) / 8U;
}

void print_parser_errors(const nvonnxparser::IParser& parser) {
    for (int i = 0; i < parser.getNbErrors(); ++i) {
        const nvonnxparser::IParserError* error = parser.getError(i);
        if (error != nullptr) {
            std::cerr << "[ONNX Parser] " << error->desc() << '\n';
        }
    }
}

bool network_has_dynamic_inputs(const nvinfer1::INetworkDefinition& network) {
    for (int32_t i = 0; i < network.getNbInputs(); ++i) {
        const nvinfer1::ITensor* input = network.getInput(i);
        if (input != nullptr && has_dynamic_dimension(input->getDimensions())) {
            return true;
        }
    }
    return false;
}

void add_single_shape_profile(nvinfer1::IBuilder& builder,
                              nvinfer1::IBuilderConfig& config,
                              const nvinfer1::INetworkDefinition& network,
                              const std::vector<InputShape>& input_shapes) {
    if (!network_has_dynamic_inputs(network)) {
        return;
    }

    // The builder owns profiles created with createOptimizationProfile().
    nvinfer1::IOptimizationProfile* profile = builder.createOptimizationProfile();
    if (profile == nullptr) {
        throw std::runtime_error("Failed to create TensorRT optimization profile.");
    }

    for (int32_t i = 0; i < network.getNbInputs(); ++i) {
        const nvinfer1::ITensor* input = network.getInput(i);
        if (input == nullptr) {
            throw std::runtime_error("TensorRT network returned a null input.");
        }

        const std::string name = input->getName();
        const nvinfer1::Dims model_dims = input->getDimensions();
        if (!has_dynamic_dimension(model_dims)) {
            continue;
        }

        const InputShape* shape = find_input_shape(input_shapes, name);
        if (shape == nullptr) {
            throw std::runtime_error("Dynamic ONNX input " + name +
                                     " requires --input-shape " + name + ":D0xD1x...");
        }

        const nvinfer1::Dims dims = to_dims(shape->dimensions);
        if (!profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims) ||
            !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims) ||
            !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims)) {
            throw std::runtime_error("Failed to set optimization profile dimensions for " + name);
        }
    }

    if (!profile->isValid()) {
        throw std::runtime_error("TensorRT optimization profile is invalid.");
    }
    if (config.addOptimizationProfile(profile) < 0) {
        throw std::runtime_error("Failed to add TensorRT optimization profile.");
    }
}

struct TimingCacheResult {
    bool loaded = false;
    bool written = false;
    std::size_t bytes = 0;
};

TensorRtPtr<nvinfer1::ITimingCache> create_timing_cache(
    nvinfer1::IBuilderConfig& builder_config,
    const std::string& path,
    TimingCacheResult* result) {
    if (path.empty()) {
        return nullptr;
    }

    const std::vector<char> cache_bytes = read_optional_binary_file(path);
    result->loaded = !cache_bytes.empty();

    TensorRtPtr<nvinfer1::ITimingCache> timing_cache{builder_config.createTimingCache(
        cache_bytes.empty() ? nullptr : cache_bytes.data(), cache_bytes.size())};
    if (!timing_cache) {
        throw std::runtime_error("Failed to create TensorRT timing cache.");
    }

    if (!builder_config.setTimingCache(*timing_cache, false)) {
        throw std::runtime_error("Failed to attach TensorRT timing cache.");
    }

    return timing_cache;
}

void serialize_timing_cache(nvinfer1::ITimingCache& timing_cache,
                            const std::string& path,
                            TimingCacheResult* result) {
    if (path.empty()) {
        return;
    }

    TensorRtPtr<nvinfer1::IHostMemory> serialized{timing_cache.serialize()};
    if (!serialized) {
        throw std::runtime_error("Failed to serialize TensorRT timing cache.");
    }

    write_binary_file(path, serialized->data(), serialized->size());
    result->written = true;
    result->bytes = serialized->size();
}

struct EngineBuildResult {
    std::vector<char> engine_bytes;
    TimingCacheResult timing_cache;
};

EngineBuildResult build_serialized_engine_from_onnx(const AppConfig& config,
                                                    TensorRtLogger& logger,
                                                    bool* tf32_allowed) {
    TensorRtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder.");
    }

    const auto strongly_typed =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    TensorRtPtr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(static_cast<nvinfer1::NetworkDefinitionCreationFlags>(
            strongly_typed))};
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network definition.");
    }

    TensorRtPtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    if (!parser) {
        throw std::runtime_error("Failed to create TensorRT ONNX parser.");
    }

    const int parser_verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    if (!parser->parseFromFile(config.onnx_path.c_str(), parser_verbosity)) {
        print_parser_errors(*parser);
        throw std::runtime_error("Failed to parse ONNX file: " + config.onnx_path);
    }

    TensorRtPtr<nvinfer1::IBuilderConfig> builder_config{builder->createBuilderConfig()};
    if (!builder_config) {
        throw std::runtime_error("Failed to create TensorRT builder config.");
    }
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                       workspace_bytes(config.workspace_mib));

    *tf32_allowed = config.allow_tf32;
    if (!config.allow_tf32) {
        builder_config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    add_single_shape_profile(*builder, *builder_config, *network, config.input_shapes);
    const std::string timing_cache_path = resolve_timing_cache_path(config);
    TimingCacheResult timing_cache_result;
    TensorRtPtr<nvinfer1::ITimingCache> timing_cache =
        create_timing_cache(*builder_config, timing_cache_path, &timing_cache_result);

    TensorRtPtr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *builder_config)};
    if (!serialized) {
        throw std::runtime_error("Failed to build serialized TensorRT engine.");
    }

    const auto* begin = static_cast<const char*>(serialized->data());
    EngineBuildResult result;
    result.engine_bytes = std::vector<char>(begin, begin + serialized->size());
    if (timing_cache) {
        serialize_timing_cache(*timing_cache, timing_cache_path, &timing_cache_result);
    }
    result.timing_cache = timing_cache_result;
    return result;
}

void apply_runtime_input_shapes(nvinfer1::IExecutionContext& context,
                                const std::vector<InputShape>& input_shapes) {
    for (const InputShape& shape : input_shapes) {
        if (!context.setInputShape(shape.tensor_name.c_str(), to_dims(shape.dimensions))) {
            throw std::runtime_error("Failed to set runtime input shape for " +
                                     shape.tensor_name);
        }
    }
}

TensorBuffer make_tensor_buffer(const nvinfer1::ICudaEngine& engine,
                                const nvinfer1::IExecutionContext& context,
                                const std::string& name) {
    const nvinfer1::Dims dims = context.getTensorShape(name.c_str());
    if (has_dynamic_dimension(dims)) {
        throw std::runtime_error("Tensor " + name +
                                 " has unresolved dynamic dimensions. Pass --input-shape.");
    }

    TensorSummary summary;
    summary.name = name;
    summary.mode = to_string(engine.getTensorIOMode(name.c_str()));
    summary.location = to_string(engine.getTensorLocation(name.c_str()));
    summary.data_type = to_string(engine.getTensorDataType(name.c_str()));
    summary.dimensions = to_vector(dims);
    summary.byte_count = checked_byte_count(engine, name, dims);

    TensorBuffer buffer(std::move(summary));
    if (engine.getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kDEVICE) {
        buffer.device = std::make_unique<DeviceBuffer>(buffer.summary.byte_count);
    }
    return buffer;
}

void prepare_inputs(std::vector<TensorBuffer>& buffers, cudaStream_t stream) {
    for (TensorBuffer& buffer : buffers) {
        if (buffer.summary.mode != "input") {
            continue;
        }

        std::fill_n(buffer.host.data(), buffer.host.byte_count(), std::uint8_t{0});
        if (buffer.device) {
            check_cuda(cudaMemcpyAsync(buffer.device->get(), buffer.host.get(),
                                       buffer.summary.byte_count, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(host-to-device input)");
        }
    }
}

void copy_outputs_to_host(std::vector<TensorBuffer>& buffers, cudaStream_t stream) {
    for (TensorBuffer& buffer : buffers) {
        if (buffer.summary.mode != "output" || !buffer.device) {
            continue;
        }
        check_cuda(cudaMemcpyAsync(buffer.host.get(), buffer.device->get(),
                                   buffer.summary.byte_count, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(device-to-host output)");
    }
}

std::uint64_t checksum_bytes(const std::uint8_t* data, std::size_t byte_count) {
    std::uint64_t checksum = 1469598103934665603ULL;
    for (std::size_t i = 0; i < byte_count; ++i) {
        checksum ^= data[i];
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

AppReport run_tensorrt_cpp_basic(const AppConfig& config) {
    if (config.engine_path.empty()) {
        throw std::runtime_error("engine_path must not be empty.");
    }
    if (!config.load_engine_only && config.onnx_path.empty()) {
        throw std::runtime_error("onnx_path must not be empty when building an engine.");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::runtime_error("Invalid warmup or measured iteration count.");
    }

    TensorRtLogger logger;
    bool tf32_allowed = false;
    TimingCacheResult timing_cache_result;
    std::vector<char> engine_bytes;

    if (config.load_engine_only) {
        engine_bytes = read_binary_file(config.engine_path);
    } else {
        EngineBuildResult build_result =
            build_serialized_engine_from_onnx(config, logger, &tf32_allowed);
        engine_bytes = std::move(build_result.engine_bytes);
        timing_cache_result = build_result.timing_cache;
        write_binary_file(config.engine_path, engine_bytes.data(), engine_bytes.size());
    }

    TensorRtPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    TensorRtPtr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size())};
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine: " + config.engine_path);
    }

    TensorRtPtr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }
    apply_runtime_input_shapes(*context, config.input_shapes);

    std::vector<TensorBuffer> buffers;
    buffers.reserve(static_cast<std::size_t>(engine->getNbIOTensors()));

    AppReport report;
    report.onnx_path = config.load_engine_only ? "" : config.onnx_path;
    report.engine_path = config.engine_path;
    report.timing_cache_path = config.load_engine_only ? "" : resolve_timing_cache_path(config);
    report.engine_built = !config.load_engine_only;
    report.tf32_allowed = tf32_allowed;
    report.timing_cache_loaded = timing_cache_result.loaded;
    report.timing_cache_written = timing_cache_result.written;
    report.engine_bytes = engine_bytes.size();
    report.timing_cache_bytes = timing_cache_result.bytes;

    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (name == nullptr) {
            throw std::runtime_error("TensorRT returned a null IO tensor name.");
        }

        TensorBuffer buffer = make_tensor_buffer(*engine, *context, name);
        if (!context->setTensorAddress(name, buffer.tensor_address())) {
            throw std::runtime_error("Failed to bind tensor address for: " + std::string(name));
        }

        if (buffer.device) {
            report.total_device_bytes += buffer.device->byte_count();
        }
        report.total_host_bytes += buffer.host.byte_count();
        buffers.push_back(std::move(buffer));
    }

    CudaStream stream;
    prepare_inputs(buffers, stream.get());
    stream.synchronize();

    for (int i = 0; i < config.warmup_iterations; ++i) {
        if (!context->enqueueV3(stream.get())) {
            throw std::runtime_error("TensorRT warmup enqueueV3 failed.");
        }
    }
    stream.synchronize();

    CudaEvent start;
    CudaEvent stop;
    check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
    for (int i = 0; i < config.measured_iterations; ++i) {
        if (!context->enqueueV3(stream.get())) {
            throw std::runtime_error("TensorRT measured enqueueV3 failed.");
        }
    }
    check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");
    report.average_enqueue_ms =
        elapsed_ms(start, stop) / static_cast<float>(config.measured_iterations);

    copy_outputs_to_host(buffers, stream.get());
    stream.synchronize();

    report.tensors.reserve(buffers.size());
    for (TensorBuffer& buffer : buffers) {
        if (buffer.summary.mode == "output") {
            buffer.summary.output_checksum =
                checksum_bytes(buffer.host.data(), buffer.host.byte_count());
        }
        report.tensors.push_back(buffer.summary);
    }

    return report;
}

}  // namespace lesson09
