#include "dynamic_batch_runner.hpp"
#include "batch_layout.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
py::dict to_dict(const lesson17::InferenceTiming& timing) {
    py::dict result;
    result["batch_size"] = timing.batch_size;
    result["output_elements"] = lesson17::checked_volume(timing.output_shape);
    result["h2d_ms"] = timing.h2d_ms;
    result["compute_ms"] = timing.compute_ms;
    result["d2h_ms"] = timing.d2h_ms;
    result["output_checksum"] = timing.output_checksum;
    return result;
}
}

PYBIND11_MODULE(trt_inference_py, module) {
    module.doc() = "Python binding for the lesson 17 TensorRT dynamic-batch runner.";
    py::class_<lesson17::DynamicBatchRunner>(module, "TensorRtSession")
        .def(py::init<const std::string&>(), py::arg("engine_path"))
        .def("infer", [](lesson17::DynamicBatchRunner& runner,
                          py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            const auto info = input.request();
            if (info.ndim != 4) {
                throw std::invalid_argument("input must be a 4-D NCHW array");
            }
            if (info.shape[1] != 3 || info.shape[2] != 640 || info.shape[3] != 640) {
                throw std::invalid_argument("input shape must be [N, 3, 640, 640]");
            }
            const auto batch = static_cast<std::size_t>(info.shape[0]);
            std::vector<float> values(static_cast<std::size_t>(info.size));
            const auto* begin = static_cast<const float*>(info.ptr);
            values.assign(begin, begin + info.size);
            lesson17::InferenceTiming timing;
            {
                py::gil_scoped_release release;
                timing = runner.infer(values, batch);
            }
            return to_dict(timing);
        }, py::arg("input"));
}
