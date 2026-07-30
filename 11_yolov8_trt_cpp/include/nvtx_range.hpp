#pragma once

#include <nvtx3/nvToolsExt.h>

#include <string>

namespace lesson11 {

// A small scope guard keeps profiler annotations paired even when an operation throws.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }

    explicit NvtxRange(const std::string& name) : NvtxRange(name.c_str()) {}

    ~NvtxRange() {
        nvtxRangePop();
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace lesson11
