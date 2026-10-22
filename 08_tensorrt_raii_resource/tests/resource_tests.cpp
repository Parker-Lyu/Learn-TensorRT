// White-box tests exercise the actual private wrappers, without exposing a new public API.
#include "../src/tensorrt_raii.cpp"

#include <atomic>
#include <type_traits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void check_traits() {
    static_assert(!std::is_copy_constructible_v<T>);
    static_assert(!std::is_copy_assignable_v<T>);
    static_assert(std::is_nothrow_move_constructible_v<T>);
    static_assert(std::is_nothrow_move_assignable_v<T>);
    static_assert(std::is_nothrow_destructible_v<T>);
}

template <typename T>
void check_buffer_moves() {
    T source(64);
    void* allocation = source.get();
    T moved(std::move(source));
    require(source.get() == nullptr && moved.get() == allocation,
            "Buffer move construction lost exclusive ownership.");
    T destination(128);
    destination = std::move(moved);
    require(moved.get() == nullptr && destination.get() == allocation,
            "Buffer move assignment lost exclusive ownership.");
    T& alias = destination;
    destination = std::move(alias);
    require(destination.get() == allocation, "Buffer self-move changed ownership.");
    source = std::move(destination);
    require(destination.get() == nullptr && source.get() == allocation,
            "Moved-from buffer could not be reassigned.");
}

template <typename T>
void check_handle_moves() {
    T source;
    auto handle = source.get();
    T moved(std::move(source));
    require(source.get() == nullptr && moved.get() == handle,
            "Handle move construction lost ownership.");
    T destination;
    destination = std::move(moved);
    require(moved.get() == nullptr && destination.get() == handle,
            "Handle move assignment lost ownership.");
    T& alias = destination;
    destination = std::move(alias);
    require(destination.get() == handle, "Handle self-move changed ownership.");
}

void CUDART_CB mark_complete(void* data) {
    static_cast<std::atomic<bool>*>(data)->store(true);
}

void check_stream_cleanup() {
    std::atomic<bool> completed{false};
    {
        lesson08::CudaStream stream;
        lesson08::check_cuda(cudaLaunchHostFunc(stream.get(), mark_complete, &completed),
                             "cudaLaunchHostFunc");
    }
    bool drained = completed.load();
    lesson08::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(test cleanup)");
    require(drained, "Stream destruction returned before work completed.");

    completed.store(false);
    try {
        lesson08::CudaStream stream;
        lesson08::check_cuda(cudaLaunchHostFunc(stream.get(), mark_complete, &completed),
                             "cudaLaunchHostFunc");
        throw std::logic_error("synthetic unwind");
    } catch (const std::logic_error&) {
        drained = completed.load();
        lesson08::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(test cleanup)");
        require(drained, "Stream did not drain during exception unwinding.");
    }

    completed.store(false);
    lesson08::CudaStream source;
    lesson08::CudaStream destination;
    lesson08::check_cuda(cudaLaunchHostFunc(destination.get(), mark_complete, &completed),
                         "cudaLaunchHostFunc");
    destination = std::move(source);
    drained = completed.load();
    lesson08::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(test cleanup)");
    require(drained, "Move assignment did not drain the old stream.");
    bool rejected = false;
    try {
        source.synchronize();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Moved-from stream synchronized the CUDA default stream.");
}

}  // namespace

int main() {
    check_traits<lesson08::DeviceBuffer>();
    check_traits<lesson08::PinnedHostBuffer>();
    check_traits<lesson08::CudaStream>();
    check_traits<lesson08::CudaEvent>();
    try {
        for (int rank : {-1, nvinfer1::Dims::MAX_DIMS + 1}) {
            nvinfer1::Dims dims{};
            dims.nbDims = rank;
            bool rejected = false;
            try {
                (void)lesson08::has_dynamic_dimension(dims);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected, "Invalid tensor rank was not rejected.");
        }
        bool rejected = false;
        try {
            lesson08::DeviceBuffer zero(0);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "Zero-byte device allocation was not rejected.");

        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0) {
            std::cerr << "SKIP: a CUDA device is required for ownership runtime tests.\n";
            return 77;
        }
        check_buffer_moves<lesson08::DeviceBuffer>();
        check_buffer_moves<lesson08::PinnedHostBuffer>();
        check_handle_moves<lesson08::CudaStream>();
        check_handle_moves<lesson08::CudaEvent>();
        check_stream_cleanup();

        lesson08::DeviceBuffer source(64);
        lesson08::DeviceBuffer moved(std::move(source));
        require(source.byte_count() == 0 && moved.byte_count() == 64,
                "Device buffer byte counts were not transferred.");
        lesson08::check_cuda(cudaMemset(moved.get(), 42, moved.byte_count()), "cudaMemset");
        unsigned char value = 0;
        lesson08::check_cuda(cudaMemcpy(&value, moved.get(), 1, cudaMemcpyDeviceToHost),
                             "cudaMemcpy");
        require(value == 42, "Moved device allocation is not usable.");
        std::cout << "Resource ownership checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
