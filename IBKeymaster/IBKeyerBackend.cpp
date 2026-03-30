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
                           int p_PrematteEnabled, int p_PrematteBlur, int p_PrematteErode, int p_PrematteIterations,
                           int p_GuidedFilterEnabled, int p_GuidedFilterMode, int p_GuidedRadius,
                           float p_GuidedEpsilon, float p_GuidedMix,
                           float p_EdgeProtect, int p_RefineIterations,
                           float p_EdgeColorCorrect,
                           int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount,
                           int p_AdditiveKeyEnabled, int p_AdditiveKeyMode,
                           float p_AdditiveKeySat, float p_AdditiveKeyAmount, int p_AdditiveKeyBlackClamp,
                           int p_ViewMode,
                           const float* p_Input, const float* p_Screen,
                           const float* p_Background,
                           const float* p_GarbageMatte, const float* p_OcclusionMatte,
                           float* p_Output);
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

bool requiresReferenceOnlyFeatures(const RenderRequest& request)
{
    // This helper started life as a safety rail while the private-branch guide/composite features
    // only existed in the CPU reference path. Once the CUDA path learned the same features, keeping
    // this list would silently strand Windows/Linux on CPU and make host-CUDA look "broken" even
    // when the device code was ready. Leaving the helper in place keeps that migration story visible.
    (void)request;
    return false;
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
    frame.background = makeImagePlaneDesc(request.backgroundImage);
    frame.garbageMatte = makeImagePlaneDesc(request.garbageMatteImage);
    frame.occlusionMatte = makeImagePlaneDesc(request.occlusionMatteImage);
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

void erodeSingle(const float* src, float* dst, int width, int height, int radius)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float minValue = 1.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = std::max(0, std::min(height - 1, y + dy));
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = std::max(0, std::min(width - 1, x + dx));
                    minValue = std::min(minValue, src[(sy * width) + sx]);
                }
            }
            dst[(y * width) + x] = minValue;
        }
    }
}

float smoothstep01(float value)
{
    const float t = clamp01(value);
    return t * t * (3.0f - 2.0f * t);
}

void buildCleanPlate(const PackedFrame& frame,
                     const IBKeyerParams& params,
                     const std::vector<float>& alpha,
                     std::vector<float>& cleanPlate,
                     std::vector<float>& cleanR,
                     std::vector<float>& cleanG,
                     std::vector<float>& cleanB,
                     std::vector<float>& scratch)
{
    const int pixelCount = frame.width * frame.height;
    cleanR.resize(pixelCount);
    cleanG.resize(pixelCount);
    cleanB.resize(pixelCount);
    cleanPlate.resize(static_cast<size_t>(pixelCount) * 4u, 0.0f);

    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float alphaSoft = smoothstep01(alpha[index]);
        cleanR[index] = frame.srcRgba[rgba + 0] * (1.0f - alphaSoft) + params.pickR * alphaSoft;
        cleanG[index] = frame.srcRgba[rgba + 1] * (1.0f - alphaSoft) + params.pickG * alphaSoft;
        cleanB[index] = frame.srcRgba[rgba + 2] * (1.0f - alphaSoft) + params.pickB * alphaSoft;
    }

    const int blurRadius = std::max(1, params.prematteBlur);
    const std::vector<float> weights = buildGaussianWeights(blurRadius);
    gaussianBlurSingle(cleanR.data(), scratch.data(), frame.width, frame.height, weights, blurRadius);
    gaussianBlurSingle(cleanG.data(), scratch.data(), frame.width, frame.height, weights, blurRadius);
    gaussianBlurSingle(cleanB.data(), scratch.data(), frame.width, frame.height, weights, blurRadius);

    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        cleanPlate[rgba + 0] = cleanR[index];
        cleanPlate[rgba + 1] = cleanG[index];
        cleanPlate[rgba + 2] = cleanB[index];
        cleanPlate[rgba + 3] = 1.0f;
    }
}

void applyExternalMatte(float* alphaBuffer,
                        float* dstRgba,
                        int pixelCount,
                        const float* matteRgba,
                        bool garbage)
{
    if (matteRgba == nullptr) {
        return;
    }
    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float matteAlpha = matteRgba[rgba + 3];
        float alpha = alphaBuffer[index];
        if (garbage) {
            alpha *= (1.0f - matteAlpha);
        } else {
            alpha = std::max(alpha, matteAlpha);
        }
        alphaBuffer[index] = clamp01(alpha);
        dstRgba[rgba + 3] = alphaBuffer[index];
    }
}

void writeAlphaDiagnostic(float* dstRgba, const float* alpha, int pixelCount)
{
    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float a = alpha[index];
        dstRgba[rgba + 0] = a;
        dstRgba[rgba + 1] = a;
        dstRgba[rgba + 2] = a;
        dstRgba[rgba + 3] = 1.0f;
    }
}

void copyDiagnosticRgba(float* dstRgba, const float* srcRgba, int pixelCount)
{
    if (srcRgba == nullptr) {
        return;
    }
    std::copy(srcRgba, srcRgba + static_cast<size_t>(pixelCount) * 4u, dstRgba);
}

void writeAlphaFromRgbaDiagnostic(float* dstRgba, int pixelCount)
{
    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float a = dstRgba[rgba + 3];
        dstRgba[rgba + 0] = a;
        dstRgba[rgba + 1] = a;
        dstRgba[rgba + 2] = a;
        dstRgba[rgba + 3] = 1.0f;
    }
}

void runScalarGuidedFilter(const PackedFrame& frame,
                           const IBKeyerParams& params,
                           std::vector<float>& rawAlpha,
                           std::vector<float>& guide,
                           std::vector<float>& meanI,
                           std::vector<float>& meanP,
                           std::vector<float>& meanIp,
                           std::vector<float>& meanII,
                           std::vector<float>& scratch,
                           float* dstRgba)
{
    const int pixelCount = frame.width * frame.height;
    const std::vector<float> gaussianWeights = buildGaussianWeights(params.guidedRadius);
    std::vector<float> savedRawAlpha = rawAlpha;
    const int numIter = std::max(1, std::min(params.refineIterations, 5));

    for (int iter = 0; iter < numIter; ++iter) {
        if (iter > 0) {
            for (int index = 0; index < pixelCount; ++index) {
                const int rgba = index * 4;
                const float alpha = rawAlpha[index];
                const float fgLum = luminance(frame.srcRgba[rgba + 0] * alpha,
                                              frame.srcRgba[rgba + 1] * alpha,
                                              frame.srcRgba[rgba + 2] * alpha);
                guide[index] = fgLum * (1.0f - params.edgeProtect) + alpha * params.edgeProtect;
            }
        }

        for (int index = 0; index < pixelCount; ++index) {
            meanI[index] = guide[index];
            meanP[index] = rawAlpha[index];
            meanIp[index] = guide[index] * rawAlpha[index];
            meanII[index] = guide[index] * guide[index];
        }

        gaussianBlurSingle(meanI.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanP.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanIp.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanII.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);

        for (int index = 0; index < pixelCount; ++index) {
            const float variance = meanII[index] - meanI[index] * meanI[index];
            const float covariance = meanIp[index] - meanI[index] * meanP[index];
            const float a = covariance / (variance + params.guidedEpsilon);
            const float b = meanP[index] - a * meanI[index];
            meanI[index] = a;
            meanP[index] = b;
        }

        gaussianBlurSingle(meanI.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanP.data(), scratch.data(), frame.width, frame.height, gaussianWeights, params.guidedRadius);

        if (iter < numIter - 1) {
            for (int index = 0; index < pixelCount; ++index) {
                rawAlpha[index] = clamp01(meanI[index] * guide[index] + meanP[index]);
            }
        }
    }

    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float guidedAlpha = clamp01(meanI[index] * guide[index] + meanP[index]);
        const float alpha = savedRawAlpha[index] * (1.0f - params.guidedMix) + guidedAlpha * params.guidedMix;
        if (params.premultiply) {
            dstRgba[rgba + 0] *= alpha;
            dstRgba[rgba + 1] *= alpha;
            dstRgba[rgba + 2] *= alpha;
        }
        dstRgba[rgba + 3] = alpha;
    }
}

void runRgbGuidedFilter(const PackedFrame& frame,
                        const IBKeyerParams& params,
                        std::vector<float>& rawAlpha,
                        std::vector<float>& scratch,
                        float* dstRgba)
{
    const int width = frame.width;
    const int height = frame.height;
    const int pixelCount = width * height;
    const int numIter = std::max(1, std::min(params.refineIterations, 5));
    const std::vector<float> gaussianWeights = buildGaussianWeights(params.guidedRadius);
    std::vector<float> meanIr(pixelCount), meanIg(pixelCount), meanIb(pixelCount), meanP(pixelCount);
    std::vector<float> irir(pixelCount), irig(pixelCount), irib(pixelCount), igig(pixelCount), igib(pixelCount), ibib(pixelCount);
    std::vector<float> irp(pixelCount), igp(pixelCount), ibp(pixelCount);
    std::vector<float> meanAr(pixelCount), meanAg(pixelCount), meanAb(pixelCount), meanB(pixelCount);
    const std::vector<float> savedRawAlpha = rawAlpha;

    for (int iter = 0; iter < numIter; ++iter) {
        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float ir = frame.srcRgba[rgba + 0];
            const float ig = frame.srcRgba[rgba + 1];
            const float ib = frame.srcRgba[rgba + 2];
            const float p = (iter == 0) ? rawAlpha[index] : rawAlpha[index];
            meanIr[index] = ir;
            meanIg[index] = ig;
            meanIb[index] = ib;
            meanP[index] = p;
            irir[index] = ir * ir;
            irig[index] = ir * ig;
            irib[index] = ir * ib;
            igig[index] = ig * ig;
            igib[index] = ig * ib;
            ibib[index] = ib * ib;
            irp[index] = ir * p;
            igp[index] = ig * p;
            ibp[index] = ib * p;
        }

        gaussianBlurSingle(meanIr.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanIg.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanIb.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanP.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(irir.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(irig.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(irib.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(igig.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(igib.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(ibib.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(irp.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(igp.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(ibp.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);

        for (int index = 0; index < pixelCount; ++index) {
            const float mIr = meanIr[index];
            const float mIg = meanIg[index];
            const float mIb = meanIb[index];
            const float mP = meanP[index];

            float s_rr = irir[index] - mIr * mIr;
            const float s_rg = irig[index] - mIr * mIg;
            const float s_rb = irib[index] - mIr * mIb;
            float s_gg = igig[index] - mIg * mIg;
            const float s_gb = igib[index] - mIg * mIb;
            float s_bb = ibib[index] - mIb * mIb;

            const float c_rp = irp[index] - mIr * mP;
            const float c_gp = igp[index] - mIg * mP;
            const float c_bp = ibp[index] - mIb * mP;

            const float trace = s_rr + s_gg + s_bb;
            const float adaptEps = params.guidedEpsilon * params.guidedEpsilon /
                                   ((trace / 3.0f) + params.guidedEpsilon + 1e-10f);
            s_rr += adaptEps;
            s_gg += adaptEps;
            s_bb += adaptEps;

            const float det = s_rr * (s_gg * s_bb - s_gb * s_gb)
                            - s_rg * (s_rg * s_bb - s_gb * s_rb)
                            + s_rb * (s_rg * s_gb - s_gg * s_rb);
            const float invDet = (std::fabs(det) > 1e-12f) ? (1.0f / det) : 0.0f;

            const float inv_rr = (s_gg * s_bb - s_gb * s_gb) * invDet;
            const float inv_rg = (s_rb * s_gb - s_rg * s_bb) * invDet;
            const float inv_rb = (s_rg * s_gb - s_rb * s_gg) * invDet;
            const float inv_gg = (s_rr * s_bb - s_rb * s_rb) * invDet;
            const float inv_gb = (s_rb * s_rg - s_rr * s_gb) * invDet;
            const float inv_bb = (s_rr * s_gg - s_rg * s_rg) * invDet;

            meanAr[index] = inv_rr * c_rp + inv_rg * c_gp + inv_rb * c_bp;
            meanAg[index] = inv_rg * c_rp + inv_gg * c_gp + inv_gb * c_bp;
            meanAb[index] = inv_rb * c_rp + inv_gb * c_gp + inv_bb * c_bp;
            meanB[index] = mP - meanAr[index] * mIr - meanAg[index] * mIg - meanAb[index] * mIb;
        }

        gaussianBlurSingle(meanAr.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanAg.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanAb.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);
        gaussianBlurSingle(meanB.data(), scratch.data(), width, height, gaussianWeights, params.guidedRadius);

        if (iter < numIter - 1) {
            for (int index = 0; index < pixelCount; ++index) {
                const int rgba = index * 4;
                const float q = meanAr[index] * frame.srcRgba[rgba + 0] +
                                meanAg[index] * frame.srcRgba[rgba + 1] +
                                meanAb[index] * frame.srcRgba[rgba + 2] +
                                meanB[index];
                rawAlpha[index] = clamp01(q);
            }
        }
    }

    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float guidedAlpha = clamp01(meanAr[index] * frame.srcRgba[rgba + 0] +
                                          meanAg[index] * frame.srcRgba[rgba + 1] +
                                          meanAb[index] * frame.srcRgba[rgba + 2] +
                                          meanB[index]);
        const float alpha = savedRawAlpha[index] * (1.0f - params.guidedMix) + guidedAlpha * params.guidedMix;
        if (params.premultiply) {
            dstRgba[rgba + 0] *= alpha;
            dstRgba[rgba + 1] *= alpha;
            dstRgba[rgba + 2] *= alpha;
        }
        dstRgba[rgba + 3] = alpha;
    }
}

void applyAdditiveKey(const PackedFrame& frame,
                      const IBKeyerParams& params,
                      const std::vector<float>& blurBgR,
                      const std::vector<float>& blurBgG,
                      const std::vector<float>& blurBgB,
                      float* dstRgba)
{
    const int pixelCount = frame.width * frame.height;
    for (int index = 0; index < pixelCount; ++index) {
        const int rgba = index * 4;
        const float alpha = dstRgba[rgba + 3];
        if (alpha > 0.999f) {
            continue;
        }

        float scrR = params.pickR;
        float scrG = params.pickG;
        float scrB = params.pickB;
        if (params.useScreenInput && frame.screenRgba != nullptr) {
            scrR = frame.screenRgba[rgba + 0];
            scrG = frame.screenRgba[rgba + 1];
            scrB = frame.screenRgba[rgba + 2];
        }

        const float srcR = frame.srcRgba[rgba + 0];
        const float srcG = frame.srcRgba[rgba + 1];
        const float srcB = frame.srcRgba[rgba + 2];

        float resR = 0.0f;
        float resG = 0.0f;
        float resB = 0.0f;
        if (params.additiveKeyMode == 0) {
            resR = srcR - scrR;
            resG = srcG - scrG;
            resB = srcB - scrB;
            const float lum = luminance(resR, resG, resB);
            resR = lum * (1.0f - params.additiveKeySaturation) + resR * params.additiveKeySaturation;
            resG = lum * (1.0f - params.additiveKeySaturation) + resG * params.additiveKeySaturation;
            resB = lum * (1.0f - params.additiveKeySaturation) + resB * params.additiveKeySaturation;
        } else {
            float fR = (scrR > 1e-6f) ? srcR / scrR : 1.0f;
            float fG = (scrG > 1e-6f) ? srcG / scrG : 1.0f;
            float fB = (scrB > 1e-6f) ? srcB / scrB : 1.0f;
            const float fLum = luminance(fR, fG, fB);
            fR = fLum * (1.0f - params.additiveKeySaturation) + fR * params.additiveKeySaturation;
            fG = fLum * (1.0f - params.additiveKeySaturation) + fG * params.additiveKeySaturation;
            fB = fLum * (1.0f - params.additiveKeySaturation) + fB * params.additiveKeySaturation;
            if (!blurBgR.empty()) {
                resR = blurBgR[index] * (fR - 1.0f);
                resG = blurBgG[index] * (fG - 1.0f);
                resB = blurBgB[index] * (fB - 1.0f);
            }
        }

        if (params.additiveKeyBlackClamp) {
            resR = std::max(resR, 0.0f);
            resG = std::max(resG, 0.0f);
            resB = std::max(resB, 0.0f);
        }

        const float weight = (1.0f - alpha) * params.additiveKeyAmount;
        dstRgba[rgba + 0] += resR * weight;
        dstRgba[rgba + 1] += resG * weight;
        dstRgba[rgba + 2] += resB * weight;
    }
}

// Moved from: the old "CPU PROCESSING — FALLBACK" section.
//
// CPU code is slower, but it is the least dependent on host-specific GPU contracts. That makes it
// the best place to preserve the algorithm "as intended" and compare GPU paths against it when
// debugging correctness regressions.
void renderCpuPacked(const IBKeyerParams& params, const PackedFrame& frame)
{
    // This section is intentionally close to the old IBKeymaster CPU fallback.
    // When I first split the plugin, I simplified this path too much and that made CPU/CUDA/Metal
    // parity harder to reason about because the "reference" path was no longer actually the old
    // algorithm. The fuller structure here is deliberate: it restores the original pass ordering.
    const int width = frame.width;
    const int height = frame.height;
    const int pixelCount = width * height;
    const bool doGF = guidedFilterActive(params);
    const bool doPrematte = params.prematteEnabled && params.prematteBlur > 0;
    const bool doBgWrap = params.bgWrapEnabled && frame.backgroundRgba != nullptr && params.bgWrapAmount > 0.0f;
    const bool doAdditive = params.additiveKeyEnabled && params.additiveKeyAmount > 0.0f;
    const bool needBgBlur = doBgWrap || (doAdditive && params.additiveKeyMode == 1 && frame.backgroundRgba != nullptr);

    std::vector<float> rawAlpha(pixelCount, 0.0f);
    std::vector<float> guide(doGF ? pixelCount : 0);
    std::vector<float> meanI(doGF ? pixelCount : 0);
    std::vector<float> meanP(doGF ? pixelCount : 0);
    std::vector<float> meanIp(doGF ? pixelCount : 0);
    std::vector<float> meanII(doGF ? pixelCount : 0);
    std::vector<float> scratch((doGF || needBgBlur || doPrematte) ? pixelCount : 1, 0.0f);
    std::vector<float> cleanPlate;
    std::vector<float> cleanR;
    std::vector<float> cleanG;
    std::vector<float> cleanB;
    std::vector<float> erodedAlpha;
    const float* activeScreen = (params.useScreenInput && frame.screenRgba != nullptr) ? frame.screenRgba : nullptr;

    auto runCorePass = [&](const float* screenRgba, bool forceScreen) {
        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float srcR = frame.srcRgba[rgba + 0];
            const float srcG = frame.srcRgba[rgba + 1];
            const float srcB = frame.srcRgba[rgba + 2];

            float scrR = params.pickR;
            float scrG = params.pickG;
            float scrB = params.pickB;
            if (forceScreen && screenRgba != nullptr) {
                scrR = screenRgba[rgba + 0];
                scrG = screenRgba[rgba + 1];
                scrB = screenRgba[rgba + 2];
            }

            const float despillRGB = despillValue(srcR, srcG, srcB, params.screenColor, params.bias, params.limit);
            const float despillScreen = despillValue(scrR, scrG, scrB, params.screenColor, params.bias, params.limit);
            const float normalized = safeDivide(despillRGB, despillScreen);
            const float spillMul = std::max(0.0f, normalized);
            const float ssR = srcR - spillMul * scrR;
            const float ssG = srcG - spillMul * scrG;
            const float ssB = srcB - spillMul * scrB;

            float alpha = clamp01(1.0f - normalized);
            if (params.nearGreyExtract && params.nearGreyAmount > 0.0f) {
                const float divR = safeDivide(ssR, srcR);
                const float divG = safeDivide(ssG, srcG);
                const float divB = safeDivide(ssB, srcB);
                const float ngeAlpha = nearGreyAlpha(divR, divG, divB, params.screenColor, params.nearGreySoftness);
                alpha = alpha + params.nearGreyAmount * ngeAlpha * (1.0f - alpha);
            }
            if (params.whiteClip > params.blackClip + 1e-6f) {
                alpha = clamp01((alpha - params.blackClip) / (params.whiteClip - params.blackClip));
            }
            if (params.matteGamma != 1.0f && alpha > 0.0f && alpha < 1.0f) {
                alpha = std::pow(alpha, params.matteGamma);
            }

            const float respillMul = std::max(0.0f, despillScreen * normalized);
            frame.dstRgba[rgba + 0] = ssR + respillMul * params.respillR;
            frame.dstRgba[rgba + 1] = ssG + respillMul * params.respillG;
            frame.dstRgba[rgba + 2] = ssB + respillMul * params.respillB;
            frame.dstRgba[rgba + 3] = alpha;
            rawAlpha[index] = alpha;

            if (doGF && params.guidedFilterMode == 0) {
                const float lum = luminance(srcR, srcG, srcB);
                guide[index] = lum * (1.0f - params.edgeProtect) + alpha * params.edgeProtect;
            }
        }
    };

    runCorePass(activeScreen, params.useScreenInput && activeScreen != nullptr);

    if (doPrematte) {
        const int iterations = std::max(1, std::min(params.prematteIterations, 5));
        std::vector<float> prematteAlpha = rawAlpha;
        erodedAlpha.resize(pixelCount);
        for (int iter = 0; iter < iterations; ++iter) {
            const float* alphaSource = prematteAlpha.data();
            if (params.prematteErode > 0) {
                erodeSingle(prematteAlpha.data(), erodedAlpha.data(), width, height, params.prematteErode);
                alphaSource = erodedAlpha.data();
            }
            buildCleanPlate(frame, params, std::vector<float>(alphaSource, alphaSource + pixelCount),
                            cleanPlate, cleanR, cleanG, cleanB, scratch);
            runCorePass(cleanPlate.data(), true);
            prematteAlpha = rawAlpha;
        }
    }

    if (params.viewMode == 2) {
        if (doPrematte && !cleanPlate.empty()) {
            copyDiagnosticRgba(frame.dstRgba, cleanPlate.data(), pixelCount);
        } else if (activeScreen != nullptr) {
            copyDiagnosticRgba(frame.dstRgba, activeScreen, pixelCount);
        }
        return;
    }

    applyExternalMatte(rawAlpha.data(), frame.dstRgba, pixelCount, frame.garbageMatteRgba, true);
    applyExternalMatte(rawAlpha.data(), frame.dstRgba, pixelCount, frame.occlusionMatteRgba, false);

    if (params.viewMode == 1) {
        writeAlphaDiagnostic(frame.dstRgba, rawAlpha.data(), pixelCount);
        return;
    }

    if (doGF) {
        if (params.guidedFilterMode == 1) {
            runRgbGuidedFilter(frame, params, rawAlpha, scratch, frame.dstRgba);
        } else {
            runScalarGuidedFilter(frame, params, rawAlpha, guide, meanI, meanP, meanIp, meanII, scratch, frame.dstRgba);
        }
    } else if (params.premultiply) {
        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float alpha = frame.dstRgba[rgba + 3];
            frame.dstRgba[rgba + 0] *= alpha;
            frame.dstRgba[rgba + 1] *= alpha;
            frame.dstRgba[rgba + 2] *= alpha;
        }
    }

    if (params.viewMode == 3) {
        writeAlphaFromRgbaDiagnostic(frame.dstRgba, pixelCount);
        return;
    }

    // This edge-colour pass was one of the features lost in the first split. It looks optional in
    // UI terms, but omitting it changes the comped edge colour in ways users absolutely notice.
    if (params.edgeColorCorrect > 0.0f) {
        const bool isPremult = params.premultiply;
        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            const float alpha = frame.dstRgba[rgba + 3];
            if (alpha <= 0.005f || alpha >= 0.995f) {
                continue;
            }

            float scrR = params.pickR;
            float scrG = params.pickG;
            float scrB = params.pickB;
            if (params.useScreenInput && frame.screenRgba != nullptr) {
                scrR = frame.screenRgba[rgba + 0];
                scrG = frame.screenRgba[rgba + 1];
                scrB = frame.screenRgba[rgba + 2];
            }

            const float srcR = frame.srcRgba[rgba + 0];
            const float srcG = frame.srcRgba[rgba + 1];
            const float srcB = frame.srcRgba[rgba + 2];
            const float invA = 1.0f / alpha;
            float fgR = (srcR - scrR * (1.0f - alpha)) * invA;
            float fgG = (srcG - scrG * (1.0f - alpha)) * invA;
            float fgB = (srcB - scrB * (1.0f - alpha)) * invA;

            fgR = std::max(-0.5f, std::min(2.0f, fgR));
            fgG = std::max(-0.5f, std::min(2.0f, fgG));
            fgB = std::max(-0.5f, std::min(2.0f, fgB));

            float curR = frame.dstRgba[rgba + 0];
            float curG = frame.dstRgba[rgba + 1];
            float curB = frame.dstRgba[rgba + 2];
            if (isPremult) {
                curR *= invA;
                curG *= invA;
                curB *= invA;
            }

            const float edgeFactor = alpha * (1.0f - alpha) * 4.0f * params.edgeColorCorrect;
            float outR = curR + (fgR - curR) * edgeFactor;
            float outG = curG + (fgG - curG) * edgeFactor;
            float outB = curB + (fgB - curB) * edgeFactor;

            if (isPremult) {
                outR *= alpha;
                outG *= alpha;
                outB *= alpha;
            }

            frame.dstRgba[rgba + 0] = outR;
            frame.dstRgba[rgba + 1] = outG;
            frame.dstRgba[rgba + 2] = outB;
        }
    }

    if (params.viewMode == 4) {
        return;
    }

    std::vector<float> bgR;
    std::vector<float> bgG;
    std::vector<float> bgB;
    if (needBgBlur) {
        const int blurRadius = std::max(1, params.bgWrapBlur);
        const std::vector<float> bgWeights = buildGaussianWeights(blurRadius);
        bgR.resize(pixelCount);
        bgG.resize(pixelCount);
        bgB.resize(pixelCount);

        for (int index = 0; index < pixelCount; ++index) {
            const int rgba = index * 4;
            bgR[index] = frame.backgroundRgba[rgba + 0];
            bgG[index] = frame.backgroundRgba[rgba + 1];
            bgB[index] = frame.backgroundRgba[rgba + 2];
        }

        gaussianBlurSingle(bgR.data(), scratch.data(), width, height, bgWeights, blurRadius);
        gaussianBlurSingle(bgG.data(), scratch.data(), width, height, bgWeights, blurRadius);
        gaussianBlurSingle(bgB.data(), scratch.data(), width, height, bgWeights, blurRadius);

        if (params.viewMode == 5) {
            for (int index = 0; index < pixelCount; ++index) {
                const int rgba = index * 4;
                frame.dstRgba[rgba + 0] = bgR[index];
                frame.dstRgba[rgba + 1] = bgG[index];
                frame.dstRgba[rgba + 2] = bgB[index];
                frame.dstRgba[rgba + 3] = 1.0f;
            }
            return;
        }

        if (doBgWrap) {
            for (int index = 0; index < pixelCount; ++index) {
                const int rgba = index * 4;
                const float alpha = frame.dstRgba[rgba + 3];
                const float wrapWeight = alpha * (1.0f - alpha) * 4.0f * params.bgWrapAmount;
                frame.dstRgba[rgba + 0] += bgR[index] * wrapWeight;
                frame.dstRgba[rgba + 1] += bgG[index] * wrapWeight;
                frame.dstRgba[rgba + 2] += bgB[index] * wrapWeight;
            }
        }
    }

    if (doAdditive) {
        applyAdditiveKey(frame, params, bgR, bgG, bgB, frame.dstRgba);
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

void packMatteWindow(const OFX::Image* image, const OfxRectI& renderWindow, std::vector<float>& packed)
{
    const int width = renderWindow.x2 - renderWindow.x1;
    const int height = renderWindow.y2 - renderWindow.y1;
    packed.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    if (image == nullptr) {
        return;
    }

    const OFX::PixelComponentEnum components = image->getPixelComponents();
    const int componentCount = (components == OFX::ePixelComponentRGBA) ? 4 : 3;

    for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
        for (int x = renderWindow.x1; x < renderWindow.x2; ++x) {
            const float* sourcePixel = static_cast<const float*>(image->getPixelAddress(x, y));
            if (sourcePixel == nullptr) {
                continue;
            }
            const int localIndex = ((y - renderWindow.y1) * width + (x - renderWindow.x1)) * 4;
            const float matte = (componentCount == 4)
                ? sourcePixel[3]
                : luminance(sourcePixel[0], sourcePixel[1], sourcePixel[2]);
            packed[localIndex + 0] = matte;
            packed[localIndex + 1] = matte;
            packed[localIndex + 2] = matte;
            packed[localIndex + 3] = matte;
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
    std::vector<float> backgroundPacked;
    std::vector<float> garbagePacked;
    std::vector<float> occlusionPacked;
    std::vector<float> dstPacked(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    packImageWindow(request.srcImage, request.renderWindow, srcPacked);
    if (request.params.useScreenInput && request.screenImage != nullptr) {
        packImageWindow(request.screenImage, request.renderWindow, screenPacked);
    }
    if (request.params.bgWrapEnabled && request.backgroundImage != nullptr) {
        packImageWindow(request.backgroundImage, request.renderWindow, backgroundPacked);
    }
    if (request.garbageMatteImage != nullptr) {
        packMatteWindow(request.garbageMatteImage, request.renderWindow, garbagePacked);
    }
    if (request.occlusionMatteImage != nullptr) {
        packMatteWindow(request.occlusionMatteImage, request.renderWindow, occlusionPacked);
    }

    PackedFrame frame;
    frame.width = width;
    frame.height = height;
    frame.srcRgba = srcPacked.data();
    frame.screenRgba = screenPacked.empty() ? nullptr : screenPacked.data();
    frame.backgroundRgba = backgroundPacked.empty() ? nullptr : backgroundPacked.data();
    frame.garbageMatteRgba = garbagePacked.empty() ? nullptr : garbagePacked.data();
    frame.occlusionMatteRgba = occlusionPacked.empty() ? nullptr : occlusionPacked.data();
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
    std::vector<float> backgroundPacked;
    std::vector<float> garbagePacked;
    std::vector<float> occlusionPacked;
    std::vector<float> dstPacked(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    packImageWindow(request.srcImage, request.renderWindow, srcPacked);
    if (request.params.useScreenInput && request.screenImage != nullptr) {
        packImageWindow(request.screenImage, request.renderWindow, screenPacked);
    }
    if (request.params.bgWrapEnabled && request.backgroundImage != nullptr) {
        packImageWindow(request.backgroundImage, request.renderWindow, backgroundPacked);
    }
    if (request.garbageMatteImage != nullptr) {
        packMatteWindow(request.garbageMatteImage, request.renderWindow, garbagePacked);
    }
    if (request.occlusionMatteImage != nullptr) {
        packMatteWindow(request.occlusionMatteImage, request.renderWindow, occlusionPacked);
    }

    PackedFrame frame;
    frame.width = width;
    frame.height = height;
    frame.srcRgba = srcPacked.data();
    frame.screenRgba = screenPacked.empty() ? nullptr : screenPacked.data();
    frame.backgroundRgba = backgroundPacked.empty() ? nullptr : backgroundPacked.data();
    frame.garbageMatteRgba = garbagePacked.empty() ? nullptr : garbagePacked.data();
    frame.occlusionMatteRgba = occlusionPacked.empty() ? nullptr : occlusionPacked.data();
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

    if (request.params.bgWrapEnabled && request.backgroundImage != nullptr) {
        if (frame.background.data == nullptr || frame.background.rowBytes == 0 || (frame.background.components != 3 && frame.background.components != 4)) {
            return {
                false,
                BackendKind::HostCUDA,
                "Host CUDA declined because the Background clip did not expose a usable RGB/RGBA device buffer."
            };
        }
    }
    if (request.garbageMatteImage != nullptr) {
        if (frame.garbageMatte.data == nullptr || frame.garbageMatte.rowBytes == 0 || (frame.garbageMatte.components != 3 && frame.garbageMatte.components != 4)) {
            return {false, BackendKind::HostCUDA, "Host CUDA declined because the Garbage Matte clip did not expose a usable RGB/RGBA device buffer."};
        }
    }
    if (request.occlusionMatteImage != nullptr) {
        if (frame.occlusionMatte.data == nullptr || frame.occlusionMatte.rowBytes == 0 || (frame.occlusionMatte.components != 3 && frame.occlusionMatte.components != 4)) {
            return {false, BackendKind::HostCUDA, "Host CUDA declined because the Occlusion Matte clip did not expose a usable RGB/RGBA device buffer."};
        }
    }

    logMessage(false, formatString(
        "IBKeyer: HostCUDA zero-copy validated. prematte=%d rgbGuide=%d bgWrap=%d additive=%d garbage=%d occlusion=%d viewMode=%d",
        request.params.prematteEnabled ? 1 : 0,
        (request.params.guidedFilterEnabled && request.params.guidedFilterMode == 1) ? 1 : 0,
        request.params.bgWrapEnabled ? 1 : 0,
        request.params.additiveKeyEnabled ? 1 : 0,
        request.garbageMatteImage != nullptr ? 1 : 0,
        request.occlusionMatteImage != nullptr ? 1 : 0,
        request.params.viewMode));

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

    if (request.params.bgWrapEnabled && request.backgroundImage != nullptr) {
        const OfxRectI& bgBounds = request.backgroundImage->getBounds();
        if (bgBounds.x1 != srcBounds.x1 ||
            bgBounds.y1 != srcBounds.y1 ||
            bgBounds.x2 != srcBounds.x2 ||
            bgBounds.y2 != srcBounds.y2) {
            return {
                false,
                BackendKind::HostMetal,
                "Host Metal requires the Background clip to match the Source bounds; mismatched host buffers fall back to CPU rather than guessing at per-host buffer layouts."
            };
        }

        if (request.backgroundImage->getPixelComponents() != OFX::ePixelComponentRGBA) {
            return {
                false,
                BackendKind::HostMetal,
                "Host Metal currently assumes the Background clip arrives as RGBA device memory; RGB Background clips fall back to CPU so the wrapper does not lie about parity."
            };
        }
    }

    auto validateMetalMatteClip = [&](const OFX::Image* image, const char* name) -> BackendResult {
        if (image == nullptr) {
            return {true, BackendKind::HostMetal, ""};
        }
        const OfxRectI& matteBounds = image->getBounds();
        if (matteBounds.x1 != srcBounds.x1 || matteBounds.y1 != srcBounds.y1 ||
            matteBounds.x2 != srcBounds.x2 || matteBounds.y2 != srcBounds.y2) {
            return {false, BackendKind::HostMetal, formatString("Host Metal requires the %s clip to match the Source bounds; mismatched host buffers fall back to CPU rather than guessing at per-host buffer layouts.", name)};
        }
        if (image->getPixelComponents() != OFX::ePixelComponentRGBA) {
            return {false, BackendKind::HostMetal, formatString("Host Metal currently assumes the %s clip arrives as RGBA device memory; RGB mattes fall back to CPU so the wrapper does not misread host Metal buffers.", name)};
        }
        return {true, BackendKind::HostMetal, ""};
    };

    if (const BackendResult matte = validateMetalMatteClip(request.garbageMatteImage, "Garbage Matte"); !matte.success) {
        return matte;
    }
    if (const BackendResult matte = validateMetalMatteClip(request.occlusionMatteImage, "Occlusion Matte"); !matte.success) {
        return matte;
    }

    // The shared Metal kernel already had support for the richer IBKeymaster control surface.
    // The cross-platform wrapper was previously hard-coding neutral placeholders here, which made
    // macOS look feature-complete in code while quietly disabling those controls in practice.
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
        request.params.nearGreyAmount,
        request.params.nearGreySoftness,
        request.params.blackClip,
        request.params.whiteClip,
        request.params.matteGamma,
        request.params.prematteEnabled ? 1 : 0,
        request.params.prematteBlur,
        request.params.prematteErode,
        request.params.prematteIterations,
        request.params.guidedFilterEnabled ? 1 : 0,
        request.params.guidedFilterMode,
        request.params.guidedRadius,
        request.params.guidedEpsilon,
        request.params.guidedMix,
        request.params.edgeProtect,
        request.params.refineIterations,
        request.params.edgeColorCorrect,
        request.params.bgWrapEnabled ? 1 : 0,
        request.params.bgWrapBlur,
        request.params.bgWrapAmount,
        request.params.additiveKeyEnabled ? 1 : 0,
        request.params.additiveKeyMode,
        request.params.additiveKeySaturation,
        request.params.additiveKeyAmount,
        request.params.additiveKeyBlackClamp ? 1 : 0,
        request.params.viewMode,
        static_cast<const float*>(request.srcImage->getPixelData()),
        (request.params.useScreenInput && request.screenImage != nullptr)
            ? static_cast<const float*>(request.screenImage->getPixelData())
            : nullptr,
        (request.params.bgWrapEnabled && request.backgroundImage != nullptr)
            ? static_cast<const float*>(request.backgroundImage->getPixelData())
            : nullptr,
        (request.garbageMatteImage != nullptr)
            ? static_cast<const float*>(request.garbageMatteImage->getPixelData())
            : nullptr,
        (request.occlusionMatteImage != nullptr)
            ? static_cast<const float*>(request.occlusionMatteImage->getPixelData())
            : nullptr,
        static_cast<float*>(request.dstImage->getPixelData()));

    return {true, BackendKind::HostMetal, "Rendered with the host Metal backend."};
}
#endif

// Moved from: the old implicit backend choice inside the processing setup.
// One of the hard lessons in multi-backend OFX code is that silent backend choice becomes very
// hard to debug. This helper makes the decision and the reason visible in one place.
BackendKind chooseBackend(const RenderRequest& request, std::string& reason)
{
    const bool needsReferenceOnly = requiresReferenceOnlyFeatures(request);

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
            // This used to branch on "reference-only" features. Once prematte, external mattes,
            // RGB guide mode, additive key, and diagnostics landed in CUDA, keeping that older
            // wording became actively misleading during zero-copy debugging.
            reason = "The host enabled OFX CUDA render and supplied a CUDA stream, so IBKeyer stays on the host-CUDA memory path.";
            return BackendKind::HostCUDA;
        }
        reason = "The host enabled OFX CUDA render but did not supply a CUDA stream. That leaves no safe CPU-readable fallback for the CUDA images.";
        return BackendKind::HostCUDA;
    }
    if (needsReferenceOnly) {
        // These newer private-branch features were ported CPU-first so the result stays trustworthy
        // while the shared CUDA implementation catches up. This is only safe when the host has not
        // already switched fetchImage() over to CUDA device memory.
        reason = "The requested feature set currently relies on the CPU reference path on Windows/Linux to preserve parity (prematte, external mattes, additive key, RGB guided filter, or diagnostic views).";
        return BackendKind::CPU;
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
