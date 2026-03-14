#pragma once

#include <string>

#include "IBKeyerBackend.h"

namespace IBKeyerCore {

#if defined(OFX_SUPPORTS_CUDARENDER)
bool renderCudaInternal(const IBKeyerParams& params, const PackedFrame& frame, std::string& error);
bool renderCudaHost(const IBKeyerParams& params,
                    const DeviceRenderFrame& frame,
                    void* hostCudaStreamOpaque,
                    std::string& error);
#else
// macOS currently keeps Metal as the host GPU backend and does not compile the CUDA unit.
// These stubs keep the backend file linkable on non-CUDA builds while still making it obvious
// in logs why any accidental CUDA route was declined.
inline bool renderCudaInternal(const IBKeyerParams&, const PackedFrame&, std::string& error)
{
    error = "CUDA backend is not compiled into this build.";
    return false;
}

inline bool renderCudaHost(const IBKeyerParams&,
                           const DeviceRenderFrame&,
                           void*,
                           std::string& error)
{
    error = "Host CUDA is not available in this build.";
    return false;
}
#endif

} // namespace IBKeyerCore
