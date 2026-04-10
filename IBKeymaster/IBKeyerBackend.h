#pragma once

#include <cstddef>
#include <string>

#include "ofxsImageEffect.h"

#include "IBKeyerShared.h"

namespace IBKeyerCore {

enum class CudaRenderMode
{
    HostPreferred,
    InternalOnly
};

enum class BackendKind
{
    CPU,
    HostCUDA,
    InternalCUDA,
    HostMetal
};

struct ImagePlaneDesc
{
    const void* data = nullptr;
    size_t rowBytes = 0;
    OfxRectI bounds = {0, 0, 0, 0};
    int components = 0;
};

struct MutableImagePlaneDesc
{
    void* data = nullptr;
    size_t rowBytes = 0;
    OfxRectI bounds = {0, 0, 0, 0};
    int components = 0;
};

struct DeviceRenderFrame
{
    ImagePlaneDesc src;
    ImagePlaneDesc screen;
    ImagePlaneDesc background;
    ImagePlaneDesc garbageMatte;
    ImagePlaneDesc occlusionMatte;
    MutableImagePlaneDesc dst;
    OfxRectI renderWindow = {0, 0, 0, 0};
};

struct RenderRequest
{
    const OFX::Image* srcImage = nullptr;
    const OFX::Image* screenImage = nullptr;
    const OFX::Image* backgroundImage = nullptr;
    const OFX::Image* garbageMatteImage = nullptr;
    const OFX::Image* occlusionMatteImage = nullptr;
    OFX::Image* dstImage = nullptr;
    OfxRectI renderWindow = {0, 0, 0, 0};
    bool hostCudaEnabled = false;
    void* hostCudaStream = nullptr;
    bool hostMetalEnabled = false;
    void* hostMetalCmdQ = nullptr;
    IBKeyerParams params;
};

struct BackendResult
{
    bool success = false;
    BackendKind backend = BackendKind::CPU;
    std::string detail;
};

BackendResult render(const RenderRequest& request);
const char* backendName(BackendKind backend);
CudaRenderMode selectedCudaRenderMode();

} // namespace IBKeyerCore
