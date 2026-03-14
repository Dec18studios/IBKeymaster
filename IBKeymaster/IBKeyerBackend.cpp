#include "IBKeyerBackend.h"

// This file holds the processing sections that used to live inside the original monolithic
// IBKeyer.cpp. Keeping the old section vocabulary here is intentional so a reader can map:
//   old "CPU PROCESSING — FALLBACK"  -> renderCpuPacked / renderCpu
//   old "SETUP AND PROCESS"          -> RenderRequest construction in IBKeyer.cpp + render()
//   old GPU dispatch choice          -> chooseBackend / renderHostCuda / renderInternalCuda

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "IBKeyerCuda.h"
#include "ofxsLog.h"

#if defined(__APPLE__)
// Declared at global scope on purpose. When this lived inside the anonymous namespace below,
// Clang treated it as an internal-linkage function declaration, which no longer matched the
// real global definition in MetalKernel.mm and caused the universal macOS link to fail.
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                           int p_ScreenColor, int p_UseScreenInput,
                           float p_PickR, float p_PickG, float p_PickB,
                           float p_Bias, float p_Limit,
                           float p_RespillR, float p_RespillG, float p_RespillB,
                           int p_Premultiply, int p_NearGreyExtract,
                           float p_NearGreyAmount, float p_NearGreySoftness,
                           float p_BlackClip, float p_WhiteClip, float p_MatteGamma,
                           int p_GuidedFilterEnabled, int p_GuidedRadius,
                           float p_GuidedEpsilon, float p_GuidedMix,
                           float p_EdgeProtect, int p_RefineIterations,
                           float p_EdgeColorCorrect,
                           int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount,
                           const float* p_Input, const float* p_Screen,
                           const float* p_Background, float* p_Output);
#endif

namespace IBKeyerCore {
namespace {

////////////////////////////////////////////////////////////////////////////////
// DIAGNOSTICS + IMAGE DESCRIPTOR HELPERS
////////////////////////////////////////////////////////////////////////////////

bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

bool debugLogEnabled()
{
    static const bool enabled = envFlagEnabled("IBKEYER_DEBUG_LOG");
    return enabled;
}

bool fileLogEnabled()
{
    static const bool enabled = envFlagEnabled("IBKEYER_FILE_LOG");
    return enabled;
}

bool hostCudaForceSyncEnabled()
{
    static const bool enabled = envFlagEnabled("IBKEYER_HOST_CUDA_FORCE_SYNC");
    return enabled;
}

// One selector controls both descriptor advertising and runtime routing.
CudaRenderMode selectedCudaRenderModeImpl()
{
    static const CudaRenderMode mode = []() {
        const char* modeVar = std::getenv("IBKEYER_CUDA_RENDER_MODE");
        if (modeVar != nullptr && modeVar[0] != '\0') {
            std::string mode(modeVar);
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            if (mode == "INTERNAL") {
                return CudaRenderMode::InternalOnly;
            }
            if (mode == "HOST" || mode == "AUTO") {
                return CudaRenderMode::HostPreferred;
            }
        }

        // Keep the legacy toggles alive so earlier troubleshooting notes and user env setups
        // do not become dead ends after the render-policy cleanup.
        if (envFlagEnabled("IBKEYER_FORCE_INTERNAL_CUDA") || envFlagEnabled("IBKEYER_DISABLE_HOST_CUDA")) {
            return CudaRenderMode::InternalOnly;
        }
        if (envFlagEnabled("IBKEYER_ENABLE_HOST_CUDA")) {
            return CudaRenderMode::HostPreferred;
        }

        // Default back to host-preferred when CUDA is compiled in. That is the whole point of this
        // pass: if the host gives us device images + stream, use them first and avoid staging.
        return CudaRenderMode::HostPreferred;
    }();
    return mode;
}

std::string formatString(const char* format, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

void logMessage(bool always, const std::string& message)
{
    if (!always && !debugLogEnabled()) {
        return;
    }
    OFX::Log::print("%s\n", message.c_str());
    std::fprintf(stderr, "%s\n", message.c_str());

    if (!fileLogEnabled()) {
        return;
    }

    // This file logger is intentionally opt-in. 
    // when host/backend routing needs proof, a dedicated log file is much easier to read than chasing
    // stderr from Resolve helper processes.
    static std::mutex logMutex;
    static bool pathInitialized = false;
    static std::filesystem::path logPath;

    if (!pathInitialized) {
        pathInitialized = true;

        const char* overridePath = std::getenv("IBKEYER_LOG_PATH");
        if (overridePath != nullptr && overridePath[0] != '\0') {
            logPath = std::filesystem::path(overridePath);
        } else {
#if defined(_WIN32)
            const char* base = std::getenv("LOCALAPPDATA");
            if (base != nullptr && base[0] != '\0') {
                logPath = std::filesystem::path(base) / "IBKeyer" / "debug.log";
            }
#else
            const char* home = std::getenv("HOME");
            if (home != nullptr && home[0] != '\0') {
                logPath = std::filesystem::path(home) / ".cache" / "IBKeyer" / "debug.log";
            }
#endif
        }

        if (!logPath.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(logPath.parent_path(), ec);
            if (ec) {
                logPath.clear();
            }
        }
    }

    if (logPath.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(logMutex);
    std::ofstream stream(logPath, std::ios::app);
    if (stream.is_open()) {
        stream << message << '\n';
    }
}

// Moved from: small bits of validation that used to be implicit inside the old processor class.
//
// We now need one place that translates OFX image objects into backend-friendly descriptors.
// That avoids repeating the same rowBytes / bounds / component assumptions in CPU, CUDA, and Metal.
int componentCountForImage(const OFX::Image* image)
{
    if (image == nullptr) {
        return 0;
    }

    switch (image->getPixelComponents()) {
        case OFX::ePixelComponentRGBA: return 4;
        case OFX::ePixelComponentRGB: return 3;
        default: return 0;
    }
}

// Moved from: the old direct getPixelData() style access.
//
// The original GPU path effectively assumed "pixel pointer + width/height" was enough. That is
// only safe when images are tightly packed and start at origin (0,0). This descriptor keeps the
// layout facts with the image so backends can address pixels correctly on hosts that use non-zero
// bounds or pitched rows.
ImagePlaneDesc makeImagePlaneDesc(const OFX::Image* image)
{
    ImagePlaneDesc desc;
    if (image == nullptr) {
        return desc;
    }

    desc.data = image->getPixelData();
    desc.bounds = image->getBounds();
    desc.components = componentCountForImage(image);

    const int rowBytes = image->getRowBytes();
    // OFX only documents negative rowBytes for CPU images. For host-CUDA we stay strict:
    // if the host reports something unexpected here, we decline zero-copy instead of
    // guessing at pointer math on device memory we do not own.
    desc.rowBytes = rowBytes > 0 ? static_cast<size_t>(rowBytes) : 0u;
    return desc;
}

// Mutable version of the image descriptor above.
//
// Source/screen images are read-only, but the destination needs write access. Keeping that split in
// the type system makes the backend code easier to follow and avoids accidental writes to inputs.
MutableImagePlaneDesc makeMutableImagePlaneDesc(OFX::Image* image)
{
    MutableImagePlaneDesc desc;
    if (image == nullptr) {
        return desc;
    }

    desc.data = image->getPixelData();
    desc.bounds = image->getBounds();
    desc.components = componentCountForImage(image);

    const int rowBytes = image->getRowBytes();
    desc.rowBytes = rowBytes > 0 ? static_cast<size_t>(rowBytes) : 0u;
    return desc;
}

// Moved from: the old "GPU DISPATCH METHODS" idea, but generalized.
//
// Host CUDA and internal staged CUDA both run the same algorithm, but they do not receive memory
// in the same shape. This struct normalizes that difference before the CUDA file sees it.
DeviceRenderFrame buildDeviceRenderFrame(const RenderRequest& request)
{
    DeviceRenderFrame frame;
    frame.src = makeImagePlaneDesc(request.srcImage);
    frame.screen = makeImagePlaneDesc(request.screenImage);
    frame.dst = makeMutableImagePlaneDesc(request.dstImage);
    frame.renderWindow = request.renderWindow;
    return frame;
}

std::vector<float> buildGaussianWeights(int radius)
{
    if (radius <= 0) {
        return {1.0f};
    }

    const int kernelSize = (radius * 2) + 1;
    const float sigma = std::max(radius / 3.0f, 0.5f);
    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

    std::vector<float> weights(static_cast<size_t>(kernelSize), 0.0f);
    float weightSum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float weight = std::exp(-(static_cast<float>(i * i)) * invTwoSigmaSq);
        weights[static_cast<size_t>(i + radius)] = weight;
        weightSum += weight;
    }

    for (float& weight : weights) {
        weight /= weightSum;
    }
    return weights;
}

void gaussianBlurSingle(float* data,
                        float* scratch,
                        int width,
                        int height,
                        const std::vector<float>& weights,
                        int radius)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int sx = std::max(0, std::min(width - 1, x + dx));
                sum += data[(y * width) + sx] * weights[static_cast<size_t>(dx + radius)];
            }
            scratch[(y * width) + x] = sum;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = std::max(0, std::min(height - 1, y + dy));
                sum += scratch[(sy * width) + x] * weights[static_cast<size_t>(dy + radius)];
            }
            data[(y * width) + x] = sum;
        }
    }
}

// Moved from: the old "CPU PROCESSING — FALLBACK" section.
//
// CPU code is slower, but it is the least dependent on host-specific GPU contracts. That makes it
// the best place to preserve the algorithm "as intended" and compare GPU paths against it when
// debugging correctness regressions.
void renderCpuPacked(const IBKeyerParams& params, const PackedFrame& frame)
{
    // This is the CPU processing fallback from the old file, moved out so it can remain the
    // reference implementation while CUDA/Metal evolve independently.
    const int width = frame.width;
    const int height = frame.height;
    const int pixelCount = width * height;
    const bool doGF = guidedFilterActive(params);

    std::vector<float> rawAlpha(doGF ? pixelCount : 0);
    std::vector<float> guide(doGF ? pixelCount : 0);
    std::vector<float> meanI(doGF ? pixelCount : 0);
    std::vector<float> meanP(doGF ? pixelCount : 0);
    std::vector<float> meanIp(doGF ? pixelCount : 0);
    std::vector<float> meanII(doGF ? pixelCount : 0);
    std::vector<float> scratch(doGF ? pixelCount : 0);
    std::vector<float> gaussianWeights(doGF ? buildGaussianWeights(params.guidedRadius) : std::vector<float>{});

    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float srcR = frame.srcRgba[rgba + 0];
        const float srcG = frame.srcRgba[rgba + 1];
        const float srcB = frame.srcRgba[rgba + 2];

        float scrR = params.pickR;
        float scrG = params.pickG;
        float scrB = params.pickB;
        if (params.useScreenInput && frame.screenRgba != nullptr) {
            scrR = frame.screenRgba[rgba + 0];
            scrG = frame.screenRgba[rgba + 1];
            scrB = frame.screenRgba[rgba + 2];
        }

        // 1. Despill of source and screen.
        const float despillRGB = despillValue(srcR, srcG, srcB, params.screenColor, params.bias, params.limit);
        const float despillScreen = despillValue(scrR, scrG, scrB, params.screenColor, params.bias, params.limit);

        // 2. Normalise.
        const float normalized = safeDivide(despillRGB, despillScreen);

        // 3. Spill map and screen subtraction.
        const float spillMul = std::max(0.0f, normalized);
        const float ssR = srcR - spillMul * scrR;
        const float ssG = srcG - spillMul * scrG;
        const float ssB = srcB - spillMul * scrB;

        // 4. Initial alpha.
        float alpha = clamp01(1.0f - normalized);

        // 5. Near Grey Extraction (optional).
        if (params.nearGreyExtract) {
            const float divR = safeDivide(ssR, srcR);
            const float divG = safeDivide(ssG, srcG);
            const float divB = safeDivide(ssB, srcB);
            const float ngeAlpha = nearGreyAlpha(divR, divG, divB, params.screenColor, params.nearGreyAmount);
            // Screen composite: a + b - a*b.
            alpha = ngeAlpha + alpha - ngeAlpha * alpha;
        }

        if (params.whiteClip > params.blackClip + 1e-6f) {
            alpha = clamp01((alpha - params.blackClip) / (params.whiteClip - params.blackClip));
        }

        // 6. Output = screen-subtracted + respill.
        const float respillMul = std::max(0.0f, despillScreen * normalized);
        frame.dstRgba[rgba + 0] = ssR + respillMul * params.respillR;
        frame.dstRgba[rgba + 1] = ssG + respillMul * params.respillG;
        frame.dstRgba[rgba + 2] = ssB + respillMul * params.respillB;
        frame.dstRgba[rgba + 3] = alpha;

        if (doGF) {
            rawAlpha[index] = alpha;
            guide[index] = luminance(srcR, srcG, srcB);
        }
    }

    if (doGF) {
        // Guided filter refinement is a post-pass layered over the original IBKeyer math.
        // Keeping it after the numbered IBK steps makes it easier to compare against the
        // pre-guided historical version when debugging parity.
        for (int index = 0; index < pixelCount; ++index) {
            meanI[index] = guide[index];
            meanP[index] = rawAlpha[index];
            meanIp[index] = guide[index] * rawAlpha[index];
            meanII[index] = guide[index] * guide[index];
        }

        // The macOS Metal path already uses a normalized Gaussian kernel for the guided
        // filter. Matching that here keeps the matte stable across OSes instead of letting
        // Windows/Linux drift because they happened to use a cheaper box-blur approximation.
        gaussianBlurSingle(meanI.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanP.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanIp.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanII.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);

        for (int index = 0; index < pixelCount; ++index) {
            const float variance = meanII[index] - meanI[index] * meanI[index];
            const float covariance = meanIp[index] - meanI[index] * meanP[index];
            const float a = covariance / (variance + params.guidedEpsilon);
            const float b = meanP[index] - a * meanI[index];
            meanI[index] = a;
            meanP[index] = b;
        }

        gaussianBlurSingle(meanI.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanP.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);

        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float raw = frame.dstRgba[rgba + 3];
            const float guided = clamp01(meanI[index] * guide[index] + meanP[index]);
            const float alpha = raw * (1.0f - params.guidedMix) + guided * params.guidedMix;
            if (params.premultiply) {
                frame.dstRgba[rgba + 0] *= alpha;
                frame.dstRgba[rgba + 1] *= alpha;
                frame.dstRgba[rgba + 2] *= alpha;
            }
            frame.dstRgba[rgba + 3] = alpha;
        }
    } else if (params.premultiply) {
        // 7. Optional premultiply.
        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float alpha = frame.dstRgba[rgba + 3];
            frame.dstRgba[rgba + 0] *= alpha;
            frame.dstRgba[rgba + 1] *= alpha;
            frame.dstRgba[rgba + 2] *= alpha;
        }
    }
}

// Moved from: the old direct CPU/GPU setup path where images were sampled ad hoc.
//
// This is the safe fallback for hosts that do not give us zero-copy CUDA buffers. Packing the
// requested window into known contiguous memory trades some performance for predictable behavior.
void packImageWindow(const OFX::Image* image, const OfxRectI& renderWindow, std::vector<float>& packed)
{
    const int width = renderWindow.x2 - renderWindow.x1;
    const int height = renderWindow.y2 - renderWindow.y1;
    packed.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    if (image == nullptr) {
        return;
    }

    const OFX::PixelComponentEnum components = image->getPixelComponents();
    const int componentCount = (components == OFX::ePixelComponentRGBA) ? 4 : 3;

    // We stage through getPixelAddress on purpose. The OFX host owns origin and stride,
    // so assuming getPixelData() is tightly packed can work on one host and silently break
    // on another. The extra copy is the price of a predictable internal CUDA path.
    for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
        for (int x = renderWindow.x1; x < renderWindow.x2; ++x) {
            const float* sourcePixel = static_cast<const float*>(image->getPixelAddress(x, y));
            if (sourcePixel == nullptr) {
                continue;
            }
            const int localIndex = ((y - renderWindow.y1) * width + (x - renderWindow.x1)) * 4;
            packed[localIndex + 0] = sourcePixel[0];
            packed[localIndex + 1] = sourcePixel[1];
            packed[localIndex + 2] = sourcePixel[2];
            packed[localIndex + 3] = (componentCount == 4) ? sourcePixel[3] : 1.0f;
        }
    }
}

// Companion to packImageWindow().
//
// Once the staged CUDA path finishes, we still need to write the result back through the host's
// own destination layout rather than assuming the host image is tightly packed.
void unpackImageWindow(const std::vector<float>& packed, const OfxRectI& renderWindow, OFX::Image* image)
{
    const int width = renderWindow.x2 - renderWindow.x1;
    for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
        for (int x = renderWindow.x1; x < renderWindow.x2; ++x) {
            float* targetPixel = static_cast<float*>(image->getPixelAddress(x, y));
            if (targetPixel == nullptr) {
                continue;
            }
            const int localIndex = ((y - renderWindow.y1) * width + (x - renderWindow.x1)) * 4;
            targetPixel[0] = packed[localIndex + 0];
            targetPixel[1] = packed[localIndex + 1];
            targetPixel[2] = packed[localIndex + 2];
            targetPixel[3] = packed[localIndex + 3];
        }
    }
}

// Moved from: the old processor setup + CPU execution flow.
//
// Keeping CPU as a named backend makes fallback logging honest and makes future parity tests much
// easier to understand.
BackendResult renderCpu(const RenderRequest& request)
{
    const int width = request.renderWindow.x2 - request.renderWindow.x1;
    const int height = request.renderWindow.y2 - request.renderWindow.y1;
    std::vector<float> srcPacked;
    std::vector<float> screenPacked;
    std::vector<float> dstPacked(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    packImageWindow(request.srcImage, request.renderWindow, srcPacked);
    if (request.params.useScreenInput && request.screenImage != nullptr) {
        packImageWindow(request.screenImage, request.renderWindow, screenPacked);
    }

    PackedFrame frame;
    frame.width = width;
    frame.height = height;
    frame.srcRgba = srcPacked.data();
    frame.screenRgba = screenPacked.empty() ? nullptr : screenPacked.data();
    frame.dstRgba = dstPacked.data();

    // CPU is intentionally kept as the reference path. When GPU behavior differs, this
    // is the path we trust first because it does not depend on host GPU contracts.
    renderCpuPacked(request.params, frame);
    unpackImageWindow(dstPacked, request.renderWindow, request.dstImage);
    return {true, BackendKind::CPU, "Rendered on CPU."};
}

////////////////////////////////////////////////////////////////////////////////
// CUDA RENDERING
////////////////////////////////////////////////////////////////////////////////

// Not every host exposes CUDA device pointers. This backend keeps Windows/Linux usable on those
// hosts without forcing the CPU path every time host interop is unavailable.
BackendResult renderInternalCuda(const RenderRequest& request)
{
    const int width = request.renderWindow.x2 - request.renderWindow.x1;
    const int height = request.renderWindow.y2 - request.renderWindow.y1;
    std::vector<float> srcPacked;
    std::vector<float> screenPacked;
    std::vector<float> dstPacked(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    packImageWindow(request.srcImage, request.renderWindow, srcPacked);
    if (request.params.useScreenInput && request.screenImage != nullptr) {
        packImageWindow(request.screenImage, request.renderWindow, screenPacked);
    }

    PackedFrame frame;
    frame.width = width;
    frame.height = height;
    frame.srcRgba = srcPacked.data();
    frame.screenRgba = screenPacked.empty() ? nullptr : screenPacked.data();
    frame.dstRgba = dstPacked.data();

    std::string error;
    if (!renderCudaInternal(request.params, frame, error)) {
        return {false, BackendKind::InternalCUDA, error};
    }

    unpackImageWindow(dstPacked, request.renderWindow, request.dstImage);
    return {true, BackendKind::InternalCUDA, "Rendered with the internal staged CUDA backend."};
}

// New: true OFX host-CUDA rendering.
//
// This is the fast path for hosts such as Resolve that can keep source and destination frames on
// the GPU and hand the plugin a CUDA stream to enqueue work onto.
//
// It refuses to "sort of" do zero-copy when the memory contract is incomplete. If rowBytes,
// components, or device pointers are not trustworthy, we fall back.
BackendResult renderHostCuda(const RenderRequest& request)
{
    // Zero-copy only makes sense if the host really handed us device pointers plus valid pitch.
    // If any of those invariants are missing, we decline here and let the staged CUDA path take
    // over rather than mixing two half-valid memory models.
    const DeviceRenderFrame frame = buildDeviceRenderFrame(request);
    if (frame.src.data == nullptr || frame.dst.data == nullptr) {
        return {
            false,
            BackendKind::HostCUDA,
            "Host CUDA requires the host to supply device pointers for Source and Output images."
        };
    }
    if (frame.src.rowBytes == 0 || frame.dst.rowBytes == 0) {
        return {
            false,
            BackendKind::HostCUDA,
            "Host CUDA requires positive rowBytes for device images so the kernel can address pitched memory safely."
        };
    }
    if (frame.src.components != 4 || frame.dst.components != 4) {
        return {
            false,
            BackendKind::HostCUDA,
            "Host CUDA currently supports RGBA Source/Output only."
        };
    }
    if (request.params.useScreenInput && request.screenImage != nullptr) {
        if (frame.screen.data == nullptr || frame.screen.rowBytes == 0 || (frame.screen.components != 3 && frame.screen.components != 4)) {
            return {
                false,
                BackendKind::HostCUDA,
                "Host CUDA declined because the Screen clip did not expose a usable RGB/RGBA device buffer."
            };
        }
    }

    std::string error;
    if (!renderCudaHost(request.params, frame, request.hostCudaStream, error)) {
        return {false, BackendKind::HostCUDA, error};
    }

    return {
        true,
        BackendKind::HostCUDA,
        hostCudaForceSyncEnabled()
            ? "Rendered with the host CUDA zero-copy backend (forced sync for diagnostics)."
            : "Rendered with the host CUDA zero-copy backend."
    };
}

#if defined(__APPLE__)
BackendResult renderHostMetal(const RenderRequest& request)
{
    const OfxRectI& srcBounds = request.srcImage->getBounds();
    const OfxRectI& dstBounds = request.dstImage->getBounds();
    const bool fullFrameRequest =
        request.renderWindow.x1 == srcBounds.x1 &&
        request.renderWindow.y1 == srcBounds.y1 &&
        request.renderWindow.x2 == srcBounds.x2 &&
        request.renderWindow.y2 == srcBounds.y2 &&
        dstBounds.x1 == srcBounds.x1 &&
        dstBounds.y1 == srcBounds.y1 &&
        dstBounds.x2 == srcBounds.x2 &&
        dstBounds.y2 == srcBounds.y2;

    if (!fullFrameRequest) {
        return {
            false,
            BackendKind::HostMetal,
            "Host Metal stays on the existing full-frame path; partial windows fall back to CPU so row origin assumptions do not leak into the port."
        };
    }

    if (request.params.useScreenInput && request.screenImage != nullptr) {
        const OfxRectI& screenBounds = request.screenImage->getBounds();
        if (screenBounds.x1 != srcBounds.x1 ||
            screenBounds.y1 != srcBounds.y1 ||
            screenBounds.x2 != srcBounds.x2 ||
            screenBounds.y2 != srcBounds.y2) {
            return {
                false,
                BackendKind::HostMetal,
                "Host Metal requires the Screen clip to match the Source bounds; mismatched host buffers fall back to CPU to keep the port deterministic."
            };
        }

        if (request.screenImage->getPixelComponents() != OFX::ePixelComponentRGBA) {
            return {
                false,
                BackendKind::HostMetal,
                "Host Metal currently assumes the Screen clip arrives as RGBA device memory; RGB Screen clips fall back to CPU so we do not misread host Metal buffers."
            };
        }
    }

    // The shared Metal kernel still expects host-provided MTLBuffer handles. We keep that
    // path only on macOS, and only when the host gave us a command queue and matching bounds.
    const float nearGreySoftness = request.params.nearGreyAmount;
    const float nearGreyAmount = request.params.nearGreyExtract ? 1.0f : 0.0f;
    const int width = srcBounds.x2 - srcBounds.x1;
    const int height = srcBounds.y2 - srcBounds.y1;

    RunMetalKernel(
        request.hostMetalCmdQ,
        width,
        height,
        request.params.screenColor,
        request.params.useScreenInput ? 1 : 0,
        request.params.pickR,
        request.params.pickG,
        request.params.pickB,
        request.params.bias,
        request.params.limit,
        request.params.respillR,
        request.params.respillG,
        request.params.respillB,
        request.params.premultiply ? 1 : 0,
        request.params.nearGreyExtract ? 1 : 0,
        nearGreyAmount,
        nearGreySoftness,
        request.params.blackClip,
        request.params.whiteClip,
        1.0f,
        request.params.guidedFilterEnabled ? 1 : 0,
        request.params.guidedRadius,
        request.params.guidedEpsilon,
        request.params.guidedMix,
        0.0f,
        1,
        0.0f,
        0,
        0,
        0.0f,
        static_cast<const float*>(request.srcImage->getPixelData()),
        (request.params.useScreenInput && request.screenImage != nullptr)
            ? static_cast<const float*>(request.screenImage->getPixelData())
            : nullptr,
        nullptr,
        static_cast<float*>(request.dstImage->getPixelData()));

    return {true, BackendKind::HostMetal, "Rendered with the host Metal backend."};
}
#endif

// Moved from: the old implicit backend choice inside the processing setup.
// One of the hard lessons in multi-backend OFX code is that silent backend choice becomes very
// hard to debug. This helper makes the decision and the reason visible in one place.
BackendKind chooseBackend(const RenderRequest& request, std::string& reason)
{
    if (envFlagEnabled("IBKEYER_FORCE_CPU")) {
        reason = "IBKEYER_FORCE_CPU forced the reference CPU path.";
        return BackendKind::CPU;
    }
#if defined(__APPLE__)
    if (request.hostMetalEnabled && request.hostMetalCmdQ != nullptr) {
        reason = "The host supplied a Metal command queue, so macOS keeps the existing host-Metal path.";
        return BackendKind::HostMetal;
    }
    reason = "The host did not provide Metal render resources, so macOS falls back to CPU.";
    return BackendKind::CPU;
#elif defined(OFX_SUPPORTS_CUDARENDER)
    const CudaRenderMode cudaMode = selectedCudaRenderMode();
    if (request.hostCudaEnabled) {
        if (request.hostCudaStream != nullptr) {
            reason = "The host enabled OFX CUDA render and supplied a CUDA stream, so IBKeyer must stay on the host-CUDA memory path.";
            return BackendKind::HostCUDA;
        }
        reason = "The host enabled OFX CUDA render but did not supply a CUDA stream. That leaves no safe CPU-readable fallback for the CUDA images.";
        return BackendKind::HostCUDA;
    }
    if (cudaMode == CudaRenderMode::HostPreferred &&
        request.hostCudaStream != nullptr) {
        reason = "Host CUDA is the selected policy and the host supplied CUDA device images plus a CUDA stream, so the zero-copy path is preferred.";
        return BackendKind::HostCUDA;
    }
    if (envFlagEnabled("IBKEYER_DISABLE_CUDA")) {
        reason = "IBKEYER_DISABLE_CUDA disabled the internal CUDA path.";
        return BackendKind::CPU;
    }
    if (cudaMode == CudaRenderMode::InternalOnly) {
        reason = "IBKEYER_CUDA_RENDER_MODE=INTERNAL (or a legacy override) forced the staged internal CUDA path.";
    } else {
        reason = "The host did not enable OFX host-CUDA rendering for this frame, so the staged internal CUDA path is the fallback.";
    }
    return BackendKind::InternalCUDA;
#else
    (void)request;
    reason = "The plugin was built without CUDA support, so CPU is the only safe backend.";
    return BackendKind::CPU;
#endif
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// PUBLIC BACKEND ROUTING
////////////////////////////////////////////////////////////////////////////////

const char* backendName(BackendKind backend)
{
    switch (backend) {
        case BackendKind::CPU: return "CPU";
        case BackendKind::HostCUDA: return "HostCUDA";
        case BackendKind::InternalCUDA: return "InternalCUDA";
        case BackendKind::HostMetal: return "HostMetal";
    }
    return "Unknown";
}

CudaRenderMode selectedCudaRenderMode()
{
    return selectedCudaRenderModeImpl();
}

// Moved from: the old "just call the processor" flow.
//
// This is now the one routing point that every render goes through. 
BackendResult render(const RenderRequest& request)
{
    // once the host has switched fetchImage() over to CUDA device memory, any "fallback"
    // that tries to read those pointers on the CPU is no longer a real fallback. 
    // it needs to be failing fast instead of falling through
    if (request.hostCudaEnabled && request.hostCudaStream == nullptr) {
        return {
            false,
            BackendKind::HostCUDA,
            "The host enabled CUDA render but did not provide a CUDA stream, so IBKeyer cannot safely read or stage those images."
        };
    }

    std::string selectionReason;
    const BackendKind preferredBackend = chooseBackend(request, selectionReason);
    logMessage(false, formatString("IBKeyer: preferred backend %s. %s",
                                   backendName(preferredBackend),
                                   selectionReason.c_str()));

    BackendResult result;
    if (preferredBackend == BackendKind::HostMetal) {
#if defined(__APPLE__)
        result = renderHostMetal(request);
        if (result.success) {
            logMessage(false, formatString("IBKeyer: rendered with %s.", backendName(result.backend)));
            return result;
        }
        logMessage(true, formatString("IBKeyer: Metal path declined, falling back to CPU. %s",
                                      result.detail.c_str()));
#endif
    } else if (preferredBackend == BackendKind::HostCUDA) {
        result = renderHostCuda(request);
        if (result.success) {
            logMessage(false, formatString("IBKeyer: rendered with %s.", backendName(result.backend)));
            return result;
        }
        if (request.hostCudaEnabled) {
            logMessage(true, formatString("IBKeyer: Host CUDA path failed while the host was supplying CUDA images. Refusing CPU staging fallback. %s",
                                          result.detail.c_str()));
            return result;
        }
        logMessage(true, formatString("IBKeyer: Host CUDA path declined, falling back to staged CUDA. %s",
                                      result.detail.c_str()));
        result = renderInternalCuda(request);
        if (result.success) {
            logMessage(false, formatString("IBKeyer: rendered with %s after host-CUDA fallback.", backendName(result.backend)));
            return result;
        }
        logMessage(true, formatString("IBKeyer: staged CUDA fallback failed, falling back to CPU. %s",
                                      result.detail.c_str()));
    } else if (preferredBackend == BackendKind::InternalCUDA) {
        result = renderInternalCuda(request);
        if (result.success) {
            logMessage(false, formatString("IBKeyer: rendered with %s.", backendName(result.backend)));
            return result;
        }
        logMessage(true, formatString("IBKeyer: internal CUDA path failed, falling back to CPU. %s",
                                      result.detail.c_str()));
    }

    result = renderCpu(request);
    if (result.success) {
        logMessage(false, "IBKeyer: rendered with CPU fallback.");
    }
    return result;
}

} // namespace IBKeyerCore
