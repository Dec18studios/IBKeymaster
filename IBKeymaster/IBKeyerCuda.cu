#include "IBKeyerCuda.h"

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)

// This file is the CUDA side of the old GPU sections from the original IBKeyer.cpp.
// The original code launched a flat RunCudaKernel() directly from the plugin file.
// The current version keeps the same algorithm, but splits the work into:
//   - host-CUDA zero-copy rendering against OFX-provided device buffers
//   - internal staged CUDA rendering when host interop is unavailable
// The comments intentionally preserve the old algorithm numbering so the refactor remains readable.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace IBKeyerCore {
namespace {

////////////////////////////////////////////////////////////////////////////////
// CUDA SCRATCH + IMAGE ADDRESSING HELPERS
////////////////////////////////////////////////////////////////////////////////

bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

bool hostCudaForceSyncEnabled()
{
    static const bool enabled = envFlagEnabled("IBKEYER_HOST_CUDA_FORCE_SYNC");
    return enabled;
}


// Once thefull guided filter moved onto CUDA, temporary GPU buffers became necessary. Caching them avoids
// paying that allocation cost every frame while still keeping the lifetime rules visible.
struct CudaScratchCache
{
    float* rawAlpha = nullptr;
    float* savedRawAlpha = nullptr;
    float* guide = nullptr;
    float* meanI = nullptr;
    float* meanP = nullptr;
    float* meanIp = nullptr;
    float* meanII = nullptr;
    float* scratch = nullptr;
    float* bgR = nullptr;
    float* bgG = nullptr;
    float* bgB = nullptr;
    float* prematteAlpha = nullptr;
    float* cleanPlate = nullptr;

    float* rgbMeanIr = nullptr;
    float* rgbMeanIg = nullptr;
    float* rgbMeanIb = nullptr;
    float* rgbMeanP = nullptr;
    float* rgbIrIr = nullptr;
    float* rgbIrIg = nullptr;
    float* rgbIrIb = nullptr;
    float* rgbIgIg = nullptr;
    float* rgbIgIb = nullptr;
    float* rgbIbIb = nullptr;
    float* rgbIrP = nullptr;
    float* rgbIgP = nullptr;
    float* rgbIbP = nullptr;
    float* rgbMeanAr = nullptr;
    float* rgbMeanAg = nullptr;
    float* rgbMeanAb = nullptr;
    float* rgbMeanB = nullptr;
    int pixelCapacity = 0;

    float* gaussianWeights = nullptr;
    int weightRadius = -1;
    size_t weightCount = 0;
    cudaEvent_t inFlightEvent = nullptr;
    cudaStream_t lastStream = nullptr;

    void release()
    {
        cudaFree(gaussianWeights);
        cudaFree(rgbMeanB);
        cudaFree(rgbMeanAb);
        cudaFree(rgbMeanAg);
        cudaFree(rgbMeanAr);
        cudaFree(rgbIbP);
        cudaFree(rgbIgP);
        cudaFree(rgbIrP);
        cudaFree(rgbIbIb);
        cudaFree(rgbIgIb);
        cudaFree(rgbIgIg);
        cudaFree(rgbIrIb);
        cudaFree(rgbIrIg);
        cudaFree(rgbIrIr);
        cudaFree(rgbMeanP);
        cudaFree(rgbMeanIb);
        cudaFree(rgbMeanIg);
        cudaFree(rgbMeanIr);
        cudaFree(cleanPlate);
        cudaFree(prematteAlpha);
        cudaFree(bgB);
        cudaFree(bgG);
        cudaFree(bgR);
        cudaFree(scratch);
        cudaFree(meanII);
        cudaFree(meanIp);
        cudaFree(meanP);
        cudaFree(meanI);
        cudaFree(guide);
        cudaFree(savedRawAlpha);
        cudaFree(rawAlpha);
        if (inFlightEvent != nullptr) {
            cudaEventDestroy(inFlightEvent);
        }

        gaussianWeights = nullptr;
        rgbMeanB = nullptr;
        rgbMeanAb = nullptr;
        rgbMeanAg = nullptr;
        rgbMeanAr = nullptr;
        rgbIbP = nullptr;
        rgbIgP = nullptr;
        rgbIrP = nullptr;
        rgbIbIb = nullptr;
        rgbIgIb = nullptr;
        rgbIgIg = nullptr;
        rgbIrIb = nullptr;
        rgbIrIg = nullptr;
        rgbIrIr = nullptr;
        rgbMeanP = nullptr;
        rgbMeanIb = nullptr;
        rgbMeanIg = nullptr;
        rgbMeanIr = nullptr;
        cleanPlate = nullptr;
        prematteAlpha = nullptr;
        bgB = nullptr;
        bgG = nullptr;
        bgR = nullptr;
        scratch = nullptr;
        meanII = nullptr;
        meanIp = nullptr;
        meanP = nullptr;
        meanI = nullptr;
        guide = nullptr;
        savedRawAlpha = nullptr;
        rawAlpha = nullptr;
        pixelCapacity = 0;
        weightRadius = -1;
        weightCount = 0;
        inFlightEvent = nullptr;
        lastStream = nullptr;
    }

    bool ensurePixelCapacity(int pixelCount, std::string& error)
    {
        if (pixelCount <= pixelCapacity) {
            return true;
        }

        cudaFree(bgB);
        cudaFree(bgG);
        cudaFree(bgR);
        cudaFree(prematteAlpha);
        cudaFree(cleanPlate);
        cudaFree(rgbMeanB);
        cudaFree(rgbMeanAb);
        cudaFree(rgbMeanAg);
        cudaFree(rgbMeanAr);
        cudaFree(rgbIbP);
        cudaFree(rgbIgP);
        cudaFree(rgbIrP);
        cudaFree(rgbIbIb);
        cudaFree(rgbIgIb);
        cudaFree(rgbIgIg);
        cudaFree(rgbIrIb);
        cudaFree(rgbIrIg);
        cudaFree(rgbIrIr);
        cudaFree(rgbMeanP);
        cudaFree(rgbMeanIb);
        cudaFree(rgbMeanIg);
        cudaFree(rgbMeanIr);
        cudaFree(scratch);
        cudaFree(meanII);
        cudaFree(meanIp);
        cudaFree(meanP);
        cudaFree(meanI);
        cudaFree(guide);
        cudaFree(savedRawAlpha);
        cudaFree(rawAlpha);

        bgB = nullptr;
        bgG = nullptr;
        bgR = nullptr;
        prematteAlpha = nullptr;
        cleanPlate = nullptr;
        rgbMeanB = nullptr;
        rgbMeanAb = nullptr;
        rgbMeanAg = nullptr;
        rgbMeanAr = nullptr;
        rgbIbP = nullptr;
        rgbIgP = nullptr;
        rgbIrP = nullptr;
        rgbIbIb = nullptr;
        rgbIgIb = nullptr;
        rgbIgIg = nullptr;
        rgbIrIb = nullptr;
        rgbIrIg = nullptr;
        rgbIrIr = nullptr;
        rgbMeanP = nullptr;
        rgbMeanIb = nullptr;
        rgbMeanIg = nullptr;
        rgbMeanIr = nullptr;
        scratch = nullptr;
        meanII = nullptr;
        meanIp = nullptr;
        meanP = nullptr;
        meanI = nullptr;
        guide = nullptr;
        savedRawAlpha = nullptr;
        rawAlpha = nullptr;

        const size_t channelBytes = static_cast<size_t>(pixelCount) * sizeof(float);
        const size_t rgbaBytes = static_cast<size_t>(pixelCount) * 4u * sizeof(float);
        if (cudaMalloc(&rawAlpha, channelBytes) != cudaSuccess ||
            cudaMalloc(&savedRawAlpha, channelBytes) != cudaSuccess ||
            cudaMalloc(&guide, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanI, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanP, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanIp, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanII, channelBytes) != cudaSuccess ||
            cudaMalloc(&scratch, channelBytes) != cudaSuccess ||
            cudaMalloc(&bgR, channelBytes) != cudaSuccess ||
            cudaMalloc(&bgG, channelBytes) != cudaSuccess ||
            cudaMalloc(&bgB, channelBytes) != cudaSuccess ||
            cudaMalloc(&prematteAlpha, channelBytes) != cudaSuccess ||
            cudaMalloc(&cleanPlate, rgbaBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanIr, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanIg, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanIb, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanP, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIrIr, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIrIg, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIrIb, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIgIg, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIgIb, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIbIb, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIrP, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIgP, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbIbP, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanAr, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanAg, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanAb, channelBytes) != cudaSuccess ||
            cudaMalloc(&rgbMeanB, channelBytes) != cudaSuccess) {
            error = "Failed to allocate CUDA scratch buffers for the guided filter.";
            release();
            return false;
        }

        pixelCapacity = pixelCount;
        return true;
    }

    bool ensureGaussianWeights(const std::vector<float>& hostWeights, int radius, std::string& error)
    {
        if (hostWeights.empty()) {
            return true;
        }
        if (gaussianWeights != nullptr && weightRadius == radius && weightCount == hostWeights.size()) {
            return true;
        }

        cudaFree(gaussianWeights);
        gaussianWeights = nullptr;
        weightRadius = -1;
        weightCount = 0;

        const size_t bytes = hostWeights.size() * sizeof(float);
        if (cudaMalloc(&gaussianWeights, bytes) != cudaSuccess) {
            error = "Failed to allocate CUDA Gaussian weights.";
            return false;
        }
        if (cudaMemcpy(gaussianWeights, hostWeights.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            error = "Failed to upload CUDA Gaussian weights.";
            cudaFree(gaussianWeights);
            gaussianWeights = nullptr;
            return false;
        }

        weightRadius = radius;
        weightCount = hostWeights.size();
        return true;
    }

    bool ensureReusable(cudaStream_t stream, bool waitForCompletion, std::string& error)
    {
        if (inFlightEvent == nullptr) {
            return true;
        }

        if (!waitForCompletion && stream != nullptr && stream == lastStream) {
            return true;
        }

        const cudaError_t syncError = cudaEventSynchronize(inFlightEvent);
        if (syncError != cudaSuccess) {
            error = std::string("Failed to fence CUDA scratch reuse: ") + cudaGetErrorString(syncError);
            return false;
        }
        return true;
    }

    bool markInFlight(cudaStream_t stream)
    {
        if (stream == nullptr) {
            return true;
        }
        if (inFlightEvent == nullptr &&
            cudaEventCreateWithFlags(&inFlightEvent, cudaEventDisableTiming) != cudaSuccess) {
            return false;
        }
        if (cudaEventRecord(inFlightEvent, stream) != cudaSuccess) {
            return false;
        }
        lastStream = stream;
        return true;
    }
};

// The host-CUDA fast path avoids source/destination copies, but it still needs temporary device
// buffers for the guided filter.
//
// Important Windows/Resolve lesson:
// a thread_local object with a non-trivial destructor means the loader may run CUDA teardown while
// unloading the plugin or tearing down OFXLoader.exe threads. That is a nasty place to call into the
// CUDA runtime and can hang plugin scanning before Resolve even opens.
//
// So we keep the per-thread reuse, but store it behind a trivial TLS pointer and intentionally let
// the OS reclaim it at process exit instead of running CUDA cleanup from a loader-sensitive destructor.
CudaScratchCache& scratchCache()
{
    thread_local CudaScratchCache* cache = nullptr;
    if (cache == nullptr) {
        cache = new CudaScratchCache();
    }
    return *cache;
}

// Moved from conceptually matching the Metal kernel's Gaussian precompute.
//
// I was first referring to the CPU path and it used a box-blur approximation, which made the matte differ by backend.
// Keeping the same Gaussian shape here is a correctness/parity fix.
std::vector<float> buildGaussianWeights(int radius)
{
    if (radius <= 0) {
        return {1.0f};
    }

    const int kernelSize = (radius * 2) + 1;
    const float sigma = fmaxf(radius / 3.0f, 0.5f);
    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

    std::vector<float> weights(static_cast<size_t>(kernelSize), 0.0f);
    float weightSum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float weight = expf(-(static_cast<float>(i * i)) * invTwoSigmaSq);
        weights[static_cast<size_t>(i + radius)] = weight;
        weightSum += weight;
    }

    for (float& weight : weights) {
        weight /= weightSum;
    }
    return weights;
}

IBKEYER_HOST_DEVICE inline bool pointInBounds(const OfxRectI& bounds, int x, int y)
{
    return x >= bounds.x1 && x < bounds.x2 && y >= bounds.y1 && y < bounds.y2;
}

// New helper 

IBKEYER_HOST_DEVICE inline const float* pixelAddress(const ImagePlaneDesc& image, int x, int y)
{
    if (image.data == nullptr || image.components <= 0 || image.rowBytes == 0 || !pointInBounds(image.bounds, x, y)) {
        return nullptr;
    }

    const char* row = reinterpret_cast<const char*>(image.data) +
        static_cast<size_t>(y - image.bounds.y1) * image.rowBytes;
    return reinterpret_cast<const float*>(row + static_cast<size_t>(x - image.bounds.x1) *
        static_cast<size_t>(image.components) * sizeof(float));
}

IBKEYER_HOST_DEVICE inline float* pixelAddress(const MutableImagePlaneDesc& image, int x, int y)
{
    if (image.data == nullptr || image.components <= 0 || image.rowBytes == 0 || !pointInBounds(image.bounds, x, y)) {
        return nullptr;
    }

    char* row = reinterpret_cast<char*>(image.data) +
        static_cast<size_t>(y - image.bounds.y1) * image.rowBytes;
    return reinterpret_cast<float*>(row + static_cast<size_t>(x - image.bounds.x1) *
        static_cast<size_t>(image.components) * sizeof(float));
}

IBKEYER_HOST_DEVICE inline void sampleRgb(const ImagePlaneDesc& image, int x, int y, float& r, float& g, float& b)
{
    const float* pixel = pixelAddress(image, x, y);
    if (pixel == nullptr) {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
        return;
    }

    r = pixel[0];
    g = pixel[1];
    b = pixel[2];
}

IBKEYER_HOST_DEVICE inline float sampleMatteValue(const ImagePlaneDesc& image, int x, int y)
{
    const float* pixel = pixelAddress(image, x, y);
    if (pixel == nullptr) {
        return 0.0f;
    }
    if (image.components >= 4) {
        return clamp01(pixel[3]);
    }
    return clamp01(luminance(pixel[0], pixel[1], pixel[2]));
}

IBKEYER_HOST_DEVICE inline float smoothstep01(float value)
{
    const float t = clamp01(value);
    return t * t * (3.0f - 2.0f * t);
}

IBKEYER_HOST_DEVICE inline void storeRgba(const MutableImagePlaneDesc& image, int x, int y, float r, float g, float b, float a)
{
    float* pixel = pixelAddress(image, x, y);
    if (pixel == nullptr) {
        return;
    }

    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
    pixel[3] = a;
}

// Moved from: the old flat IBKeyer CUDA kernel.
//
// The algorithm is the same, but the kernel now reads/writes through image descriptors so it can
// operate on either host-owned CUDA buffers or our staged fallback buffers.
__global__ void coreKernel(IBKeyerParams params,
                           ImagePlaneDesc src,
                           ImagePlaneDesc screen,
                           MutableImagePlaneDesc dst,
                           int renderX1,
                           int renderY1,
                           int width,
                           int height,
                           float* rawAlpha,
                           float* guide)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;

    float srcR = 0.0f;
    float srcG = 0.0f;
    float srcB = 0.0f;
    sampleRgb(src, imageX, imageY, srcR, srcG, srcB);

    float scrR = params.pickR;
    float scrG = params.pickG;
    float scrB = params.pickB;
    if (params.useScreenInput && screen.data != nullptr) {
        sampleRgb(screen, imageX, imageY, scrR, scrG, scrB);
    }

    // 1. Despill of source and screen.
    const float despillRGB = despillValue(srcR, srcG, srcB, params.screenColor, params.bias, params.limit);
    const float despillScreen = despillValue(scrR, scrG, scrB, params.screenColor, params.bias, params.limit);

    // 2. Normalise.
    const float normalized = safeDivide(despillRGB, despillScreen);

    // 3. Spill map and screen subtraction.
    const float spillMul = normalized > 0.0f ? normalized : 0.0f;
    const float ssR = srcR - spillMul * scrR;
    const float ssG = srcG - spillMul * scrG;
    const float ssB = srcB - spillMul * scrB;

    // 4. Initial alpha.
    float alpha = clamp01(1.0f - normalized);

    // 5. Near Grey Extraction (optional).
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
        alpha = powf(alpha, params.matteGamma);
    }

    // 6. Output = screen-subtracted + respill.
    const float respillMul = despillScreen * normalized > 0.0f ? despillScreen * normalized : 0.0f;
    storeRgba(dst,
              imageX,
              imageY,
              ssR + respillMul * params.respillR,
              ssG + respillMul * params.respillG,
              ssB + respillMul * params.respillB,
              alpha);

    rawAlpha[pixelIndex] = alpha;
    const float lum = luminance(srcR, srcG, srcB);
    guide[pixelIndex] = lum * (1.0f - params.edgeProtect) + alpha * params.edgeProtect;
}

__global__ void computeProductsKernel(int pixelCount,
                                      const float* rawAlpha,
                                      const float* guide,
                                      float* meanI,
                                      float* meanP,
                                      float* meanIp,
                                      float* meanII)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }

    const float I = guide[index];
    const float P = rawAlpha[index];
    meanI[index] = I;
    meanP[index] = P;
    meanIp[index] = I * P;
    meanII[index] = I * I;
}

__global__ void copyBufferKernel(int pixelCount, const float* src, float* dst)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }
    dst[index] = src[index];
}

__global__ void gaussianBlurHorizontalKernel(int width,
                                             int height,
                                             int radius,
                                             const float* weights,
                                             const float* src,
                                             float* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        const int sx = max(0, min(width - 1, x + dx));
        sum += src[(y * width) + sx] * weights[dx + radius];
    }
    dst[(y * width) + x] = sum;
}

__global__ void gaussianBlurVerticalKernel(int width,
                                           int height,
                                           int radius,
                                           const float* weights,
                                           const float* src,
                                           float* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int sy = max(0, min(height - 1, y + dy));
        sum += src[(sy * width) + x] * weights[dy + radius];
    }
    dst[(y * width) + x] = sum;
}

__global__ void guidedCoeffKernel(int pixelCount,
                                  float epsilon,
                                  float* meanI,
                                  float* meanP,
                                  const float* meanIp,
                                  const float* meanII)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }

    const float variance = meanII[index] - meanI[index] * meanI[index];
    const float covariance = meanIp[index] - meanI[index] * meanP[index];
    const float a = covariance / (variance + epsilon);
    const float b = meanP[index] - a * meanI[index];
    meanI[index] = a;
    meanP[index] = b;
}

__global__ void refineGuideKernel(int width,
                                  int height,
                                  ImagePlaneDesc src,
                                  int renderX1,
                                  int renderY1,
                                  float edgeProtect,
                                  const float* alphaBuffer,
                                  float* guideBuffer)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    const float alpha = alphaBuffer[pixelIndex];

    float srcR = 0.0f;
    float srcG = 0.0f;
    float srcB = 0.0f;
    sampleRgb(src, imageX, imageY, srcR, srcG, srcB);

    const float fgLum = luminance(srcR * alpha, srcG * alpha, srcB * alpha);
    guideBuffer[pixelIndex] = fgLum * (1.0f - edgeProtect) + alpha * edgeProtect;
}

__global__ void guidedEvalKernel(int pixelCount,
                                 const float* guide,
                                 const float* meanA,
                                 const float* meanB,
                                 float* dst)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }
    dst[index] = clamp01(meanA[index] * guide[index] + meanB[index]);
}

__global__ void guidedApplyKernel(IBKeyerParams params,
                                  MutableImagePlaneDesc dst,
                                  int renderX1,
                                  int renderY1,
                                  int width,
                                  int height,
                                  const float* rawAlpha,
                                  const float* guide,
                                  const float* meanA,
                                  const float* meanB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    const float guidedAlpha = clamp01(meanA[pixelIndex] * guide[pixelIndex] + meanB[pixelIndex]);
    const float alpha = rawAlpha[pixelIndex] * (1.0f - params.guidedMix) + guidedAlpha * params.guidedMix;

    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }

    if (params.premultiply) {
        pixel[0] *= alpha;
        pixel[1] *= alpha;
        pixel[2] *= alpha;
    }
    pixel[3] = alpha;
}

__global__ void edgeColorCorrectKernel(IBKeyerParams params,
                                       ImagePlaneDesc src,
                                       ImagePlaneDesc screen,
                                       MutableImagePlaneDesc dst,
                                       int renderX1,
                                       int renderY1,
                                       int width,
                                       int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }

    const float alpha = pixel[3];
    if (alpha <= 0.005f || alpha >= 0.995f) {
        return;
    }

    float srcR = 0.0f;
    float srcG = 0.0f;
    float srcB = 0.0f;
    sampleRgb(src, imageX, imageY, srcR, srcG, srcB);

    float scrR = params.pickR;
    float scrG = params.pickG;
    float scrB = params.pickB;
    if (params.useScreenInput && screen.data != nullptr) {
        sampleRgb(screen, imageX, imageY, scrR, scrG, scrB);
    }

    const float invA = 1.0f / alpha;
    float fgR = (srcR - scrR * (1.0f - alpha)) * invA;
    float fgG = (srcG - scrG * (1.0f - alpha)) * invA;
    float fgB = (srcB - scrB * (1.0f - alpha)) * invA;
    fgR = fmaxf(-0.5f, fminf(2.0f, fgR));
    fgG = fmaxf(-0.5f, fminf(2.0f, fgG));
    fgB = fmaxf(-0.5f, fminf(2.0f, fgB));

    float curR = pixel[0];
    float curG = pixel[1];
    float curB = pixel[2];
    if (params.premultiply) {
        curR *= invA;
        curG *= invA;
        curB *= invA;
    }

    const float edgeFactor = alpha * (1.0f - alpha) * 4.0f * params.edgeColorCorrect;
    float outR = curR + (fgR - curR) * edgeFactor;
    float outG = curG + (fgG - curG) * edgeFactor;
    float outB = curB + (fgB - curB) * edgeFactor;
    if (params.premultiply) {
        outR *= alpha;
        outG *= alpha;
        outB *= alpha;
    }

    pixel[0] = outR;
    pixel[1] = outG;
    pixel[2] = outB;
}

__global__ void premultiplyKernel(MutableImagePlaneDesc dst,
                                  int renderX1,
                                  int renderY1,
                                  int width,
                                  int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }

    const float alpha = pixel[3];
    pixel[0] *= alpha;
    pixel[1] *= alpha;
    pixel[2] *= alpha;
}

__global__ void extractBackgroundChannelsKernel(int width,
                                                int height,
                                                ImagePlaneDesc background,
                                                int renderX1,
                                                int renderY1,
                                                float* outR,
                                                float* outG,
                                                float* outB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;

    float bgR = 0.0f;
    float bgG = 0.0f;
    float bgB = 0.0f;
    sampleRgb(background, imageX, imageY, bgR, bgG, bgB);
    outR[pixelIndex] = bgR;
    outG[pixelIndex] = bgG;
    outB[pixelIndex] = bgB;
}

__global__ void bgWrapKernel(MutableImagePlaneDesc dst,
                             int renderX1,
                             int renderY1,
                             int width,
                             int height,
                             float amount,
                             const float* bgR,
                             const float* bgG,
                             const float* bgB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }

    const float alpha = pixel[3];
    const float wrapWeight = alpha * (1.0f - alpha) * 4.0f * amount;
    pixel[0] += bgR[pixelIndex] * wrapWeight;
    pixel[1] += bgG[pixelIndex] * wrapWeight;
    pixel[2] += bgB[pixelIndex] * wrapWeight;
}

__global__ void applyMatteKernel(float* rawAlpha,
                                 MutableImagePlaneDesc dst,
                                 ImagePlaneDesc matte,
                                 int renderX1,
                                 int renderY1,
                                 int width,
                                 int height,
                                 int mode)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    float* dstPixel = pixelAddress(dst, imageX, imageY);
    if (dstPixel == nullptr) {
        return;
    }

    const float matteValue = sampleMatteValue(matte, imageX, imageY);
    float alpha = rawAlpha[pixelIndex];
    if (mode == 0) {
        alpha *= (1.0f - matteValue);
    } else {
        alpha = fmaxf(alpha, matteValue);
    }
    alpha = clamp01(alpha);
    rawAlpha[pixelIndex] = alpha;
    dstPixel[3] = alpha;
}

__global__ void writeAlphaDiagnosticKernel(const float* rawAlpha,
                                           MutableImagePlaneDesc dst,
                                           int renderX1,
                                           int renderY1,
                                           int width,
                                           int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    const float a = rawAlpha[pixelIndex];
    storeRgba(dst, imageX, imageY, a, a, a, 1.0f);
}

__global__ void extractOutputAlphaDiagnosticKernel(MutableImagePlaneDesc dst,
                                                   int renderX1,
                                                   int renderY1,
                                                   int width,
                                                   int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }

    const float a = pixel[3];
    pixel[0] = a;
    pixel[1] = a;
    pixel[2] = a;
    pixel[3] = 1.0f;
}

__global__ void copyImageKernel(ImagePlaneDesc src,
                                MutableImagePlaneDesc dst,
                                int renderX1,
                                int renderY1,
                                int width,
                                int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const float* srcPixel = pixelAddress(src, imageX, imageY);
    float* dstPixel = pixelAddress(dst, imageX, imageY);
    if (srcPixel == nullptr || dstPixel == nullptr) {
        return;
    }

    dstPixel[0] = srcPixel[0];
    dstPixel[1] = srcPixel[1];
    dstPixel[2] = srcPixel[2];
    dstPixel[3] = (src.components >= 4) ? srcPixel[3] : 1.0f;
}

__global__ void packRgbKernel(MutableImagePlaneDesc dst,
                              int renderX1,
                              int renderY1,
                              int width,
                              int height,
                              const float* srcR,
                              const float* srcG,
                              const float* srcB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int pixelIndex = y * width + x;
    storeRgba(dst, renderX1 + x, renderY1 + y, srcR[pixelIndex], srcG[pixelIndex], srcB[pixelIndex], 1.0f);
}

__global__ void erodeAlphaKernel(const float* src,
                                 float* dst,
                                 int width,
                                 int height,
                                 int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float minValue = 1.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int sy = max(0, min(height - 1, y + dy));
        for (int dx = -radius; dx <= radius; ++dx) {
            const int sx = max(0, min(width - 1, x + dx));
            minValue = fminf(minValue, src[sy * width + sx]);
        }
    }
    dst[y * width + x] = minValue;
}

__global__ void cleanPlateEstimateKernel(ImagePlaneDesc src,
                                         int renderX1,
                                         int renderY1,
                                         int width,
                                         int height,
                                         float pickR,
                                         float pickG,
                                         float pickB,
                                         const float* alpha,
                                         float* outR,
                                         float* outG,
                                         float* outB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    float srcR = 0.0f;
    float srcG = 0.0f;
    float srcB = 0.0f;
    sampleRgb(src, imageX, imageY, srcR, srcG, srcB);
    const float t = smoothstep01(alpha[pixelIndex]);
    outR[pixelIndex] = srcR * (1.0f - t) + pickR * t;
    outG[pixelIndex] = srcG * (1.0f - t) + pickG * t;
    outB[pixelIndex] = srcB * (1.0f - t) + pickB * t;
}

__global__ void packCleanPlateKernel(int width,
                                     int height,
                                     const float* srcR,
                                     const float* srcG,
                                     const float* srcB,
                                     float* cleanPlate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int pixelIndex = y * width + x;
    const int rgba = pixelIndex * 4;
    cleanPlate[rgba + 0] = srcR[pixelIndex];
    cleanPlate[rgba + 1] = srcG[pixelIndex];
    cleanPlate[rgba + 2] = srcB[pixelIndex];
    cleanPlate[rgba + 3] = 1.0f;
}

__global__ void rgbComputeProductsKernel(ImagePlaneDesc src,
                                         int renderX1,
                                         int renderY1,
                                         int width,
                                         int height,
                                         const float* rawAlpha,
                                         float* meanIr,
                                         float* meanIg,
                                         float* meanIb,
                                         float* meanP,
                                         float* irir,
                                         float* irig,
                                         float* irib,
                                         float* igig,
                                         float* igib,
                                         float* ibib,
                                         float* irp,
                                         float* igp,
                                         float* ibp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int pixelIndex = y * width + x;
    float ir = 0.0f;
    float ig = 0.0f;
    float ib = 0.0f;
    sampleRgb(src, renderX1 + x, renderY1 + y, ir, ig, ib);
    const float p = rawAlpha[pixelIndex];
    meanIr[pixelIndex] = ir;
    meanIg[pixelIndex] = ig;
    meanIb[pixelIndex] = ib;
    meanP[pixelIndex] = p;
    irir[pixelIndex] = ir * ir;
    irig[pixelIndex] = ir * ig;
    irib[pixelIndex] = ir * ib;
    igig[pixelIndex] = ig * ig;
    igib[pixelIndex] = ig * ib;
    ibib[pixelIndex] = ib * ib;
    irp[pixelIndex] = ir * p;
    igp[pixelIndex] = ig * p;
    ibp[pixelIndex] = ib * p;
}

__global__ void rgbGuidedCoeffKernel(int pixelCount,
                                     float epsilon,
                                     const float* meanIr,
                                     const float* meanIg,
                                     const float* meanIb,
                                     const float* meanP,
                                     const float* irir,
                                     const float* irig,
                                     const float* irib,
                                     const float* igig,
                                     const float* igib,
                                     const float* ibib,
                                     const float* irp,
                                     const float* igp,
                                     const float* ibp,
                                     float* outAr,
                                     float* outAg,
                                     float* outAb,
                                     float* outB)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }

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
    const float adaptEps = epsilon * epsilon / ((trace / 3.0f) + epsilon + 1e-10f);
    s_rr += adaptEps;
    s_gg += adaptEps;
    s_bb += adaptEps;

    const float det = s_rr * (s_gg * s_bb - s_gb * s_gb)
                    - s_rg * (s_rg * s_bb - s_gb * s_rb)
                    + s_rb * (s_rg * s_gb - s_gg * s_rb);
    const float invDet = (fabsf(det) > 1e-12f) ? (1.0f / det) : 0.0f;

    const float inv_rr = (s_gg * s_bb - s_gb * s_gb) * invDet;
    const float inv_rg = (s_rb * s_gb - s_rg * s_bb) * invDet;
    const float inv_rb = (s_rg * s_gb - s_rb * s_gg) * invDet;
    const float inv_gg = (s_rr * s_bb - s_rb * s_rb) * invDet;
    const float inv_gb = (s_rb * s_rg - s_rr * s_gb) * invDet;
    const float inv_bb = (s_rr * s_gg - s_rg * s_rg) * invDet;

    const float ar = inv_rr * c_rp + inv_rg * c_gp + inv_rb * c_bp;
    const float ag = inv_rg * c_rp + inv_gg * c_gp + inv_gb * c_bp;
    const float ab = inv_rb * c_rp + inv_gb * c_gp + inv_bb * c_bp;
    outAr[index] = ar;
    outAg[index] = ag;
    outAb[index] = ab;
    outB[index] = mP - ar * mIr - ag * mIg - ab * mIb;
}

__global__ void rgbGuidedEvalKernel(ImagePlaneDesc src,
                                    int renderX1,
                                    int renderY1,
                                    int width,
                                    int height,
                                    const float* meanAr,
                                    const float* meanAg,
                                    const float* meanAb,
                                    const float* meanB,
                                    float* outAlpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int pixelIndex = y * width + x;
    float ir = 0.0f;
    float ig = 0.0f;
    float ib = 0.0f;
    sampleRgb(src, renderX1 + x, renderY1 + y, ir, ig, ib);
    outAlpha[pixelIndex] = clamp01(meanAr[pixelIndex] * ir +
                                   meanAg[pixelIndex] * ig +
                                   meanAb[pixelIndex] * ib +
                                   meanB[pixelIndex]);
}

__global__ void rgbGuidedApplyKernel(IBKeyerParams params,
                                     ImagePlaneDesc src,
                                     MutableImagePlaneDesc dst,
                                     int renderX1,
                                     int renderY1,
                                     int width,
                                     int height,
                                     const float* rawAlpha,
                                     const float* meanAr,
                                     const float* meanAg,
                                     const float* meanAb,
                                     const float* meanB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    float ir = 0.0f;
    float ig = 0.0f;
    float ib = 0.0f;
    sampleRgb(src, imageX, imageY, ir, ig, ib);
    const float guidedAlpha = clamp01(meanAr[pixelIndex] * ir +
                                      meanAg[pixelIndex] * ig +
                                      meanAb[pixelIndex] * ib +
                                      meanB[pixelIndex]);
    const float alpha = rawAlpha[pixelIndex] * (1.0f - params.guidedMix) + guidedAlpha * params.guidedMix;

    float* pixel = pixelAddress(dst, imageX, imageY);
    if (pixel == nullptr) {
        return;
    }
    if (params.premultiply) {
        pixel[0] *= alpha;
        pixel[1] *= alpha;
        pixel[2] *= alpha;
    }
    pixel[3] = alpha;
}

__global__ void additiveKeyKernel(IBKeyerParams params,
                                  ImagePlaneDesc src,
                                  ImagePlaneDesc screen,
                                  MutableImagePlaneDesc dst,
                                  int renderX1,
                                  int renderY1,
                                  int width,
                                  int height,
                                  const float* blurBgR,
                                  const float* blurBgG,
                                  const float* blurBgB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int imageX = renderX1 + x;
    const int imageY = renderY1 + y;
    const int pixelIndex = y * width + x;
    float* outPixel = pixelAddress(dst, imageX, imageY);
    if (outPixel == nullptr) {
        return;
    }

    const float alpha = outPixel[3];
    if (alpha > 0.999f) {
        return;
    }

    float srcR = 0.0f;
    float srcG = 0.0f;
    float srcB = 0.0f;
    sampleRgb(src, imageX, imageY, srcR, srcG, srcB);

    float scrR = params.pickR;
    float scrG = params.pickG;
    float scrB = params.pickB;
    if (params.useScreenInput && screen.data != nullptr) {
        sampleRgb(screen, imageX, imageY, scrR, scrG, scrB);
    }

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
    } else if (blurBgR != nullptr && blurBgG != nullptr && blurBgB != nullptr) {
        float fR = (scrR > 1e-6f) ? srcR / scrR : 1.0f;
        float fG = (scrG > 1e-6f) ? srcG / scrG : 1.0f;
        float fB = (scrB > 1e-6f) ? srcB / scrB : 1.0f;
        const float fLum = luminance(fR, fG, fB);
        fR = fLum * (1.0f - params.additiveKeySaturation) + fR * params.additiveKeySaturation;
        fG = fLum * (1.0f - params.additiveKeySaturation) + fG * params.additiveKeySaturation;
        fB = fLum * (1.0f - params.additiveKeySaturation) + fB * params.additiveKeySaturation;
        resR = blurBgR[pixelIndex] * (fR - 1.0f);
        resG = blurBgG[pixelIndex] * (fG - 1.0f);
        resB = blurBgB[pixelIndex] * (fB - 1.0f);
    }

    if (params.additiveKeyBlackClamp) {
        resR = fmaxf(resR, 0.0f);
        resG = fmaxf(resG, 0.0f);
        resB = fmaxf(resB, 0.0f);
    }

    const float weight = (1.0f - alpha) * params.additiveKeyAmount;
    outPixel[0] += resR * weight;
    outPixel[1] += resG * weight;
    outPixel[2] += resB * weight;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA LAUNCH HELPERS
////////////////////////////////////////////////////////////////////////////////

// New helper
//
// Host CUDA wants asynchronous stream usage, while the staged fallback still wants a completed
// frame before returning. This helper lets both modes share the same kernels without hiding where
// synchronization happens.
bool captureKernelStage(const char* stage, cudaStream_t stream, bool waitForCompletion, std::string& error)
{
    const cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        error = std::string(stage) + " launch failed: " + cudaGetErrorString(launchError);
        return false;
    }

    if (!waitForCompletion) {
        return true;
    }

    const cudaError_t syncError = (stream != nullptr) ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        error = std::string(stage) + " execution failed: " + cudaGetErrorString(syncError);
        return false;
    }

    return true;
}

bool runGaussianBlur(float* buffer,
                     float* scratch,
                     const float* weights,
                     int width,
                     int height,
                     int radius,
                     cudaStream_t stream,
                     bool waitForCompletion,
                     std::string& error)
{
    const dim3 threads(16, 16, 1);
    const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);

    gaussianBlurHorizontalKernel<<<blocks, threads, 0, stream>>>(width, height, radius, weights, buffer, scratch);
    if (!captureKernelStage("gaussian blur horizontal", stream, waitForCompletion, error)) {
        return false;
    }

    gaussianBlurVerticalKernel<<<blocks, threads, 0, stream>>>(width, height, radius, weights, scratch, buffer);
    if (!captureKernelStage("gaussian blur vertical", stream, waitForCompletion, error)) {
        return false;
    }

    return true;
}

// New shared CUDA executor.
//
// Before the zero-copy work, the staged CUDA path owned its entire memory model, so the kernel
// launcher and the staging code were tightly coupled. This function separates "how the algorithm
// runs on device" from "where the device pointers came from".
bool renderCudaFrame(const IBKeyerParams& params,
                     const DeviceRenderFrame& frame,
                     cudaStream_t stream,
                     bool waitForCompletion,
                     std::string& error)
{
    CudaScratchCache& scratch = scratchCache();
    const int width = frame.renderWindow.x2 - frame.renderWindow.x1;
    const int height = frame.renderWindow.y2 - frame.renderWindow.y1;
    if (width <= 0 || height <= 0 || frame.src.data == nullptr || frame.dst.data == nullptr) {
        error = "Invalid device frame passed to CUDA.";
        return false;
    }

    const int pixelCount = width * height;
    const bool doGF = guidedFilterActive(params);
    const bool doRgbGF = doGF && params.guidedFilterMode == 1;
    const bool doPrematte = params.prematteEnabled && params.prematteBlur > 0;
    const bool doBgWrap = params.bgWrapEnabled && frame.background.data != nullptr && params.bgWrapAmount > 0.0f;
    const bool doAdditive = params.additiveKeyEnabled && params.additiveKeyAmount > 0.0f;
    const bool needBgBlur = doBgWrap || (doAdditive && params.additiveKeyMode == 1 && frame.background.data != nullptr);
    if (!scratch.ensurePixelCapacity(pixelCount, error)) {
        return false;
    }
    if (!scratch.ensureReusable(stream, waitForCompletion, error)) {
        return false;
    }
    if (doGF) {
        if (!scratch.ensureGaussianWeights(buildGaussianWeights(params.guidedRadius), params.guidedRadius, error)) {
            return false;
        }
    }

    // The kernel now addresses OFX image memory through explicit bounds + rowBytes.
    // That extra bookkeeping is what lets the same code run against:
    //   1. host-provided CUDA device images (zero-copy), and
    //   2. our own temporary packed CUDA buffers (staged fallback).
    const dim3 pixelThreads(16, 16, 1);
    const dim3 pixelBlocks((width + pixelThreads.x - 1) / pixelThreads.x,
                           (height + pixelThreads.y - 1) / pixelThreads.y,
                           1);
    const int flatThreads = 256;
    const int flatBlocks = (pixelCount + flatThreads - 1) / flatThreads;

    coreKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(params,
                                                         frame.src,
                                                         frame.screen,
                                                         frame.dst,
                                                         frame.renderWindow.x1,
                                                         frame.renderWindow.y1,
                                                         width,
                                                         height,
                                                         scratch.rawAlpha,
                                                         scratch.guide);
    if (!captureKernelStage("core kernel", stream, waitForCompletion, error)) {
        return false;
    }

    if (doPrematte) {
        const int prematteIterations = max(1, min(params.prematteIterations, 5));
        if (!scratch.ensureGaussianWeights(buildGaussianWeights(max(1, params.prematteBlur)), max(1, params.prematteBlur), error)) {
            return false;
        }

        ImagePlaneDesc cleanScreen;
        cleanScreen.data = scratch.cleanPlate;
        cleanScreen.rowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
        cleanScreen.bounds = frame.renderWindow;
        cleanScreen.components = 4;

        for (int iter = 0; iter < prematteIterations; ++iter) {
            const float* alphaSource = scratch.rawAlpha;
            if (params.prematteErode > 0) {
                erodeAlphaKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                    scratch.rawAlpha,
                    scratch.prematteAlpha,
                    width,
                    height,
                    params.prematteErode);
                if (!captureKernelStage("prematte erode", stream, waitForCompletion, error)) {
                    return false;
                }
                alphaSource = scratch.prematteAlpha;
            }

            cleanPlateEstimateKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                frame.src,
                frame.renderWindow.x1,
                frame.renderWindow.y1,
                width,
                height,
                params.pickR,
                params.pickG,
                params.pickB,
                alphaSource,
                scratch.bgR,
                scratch.bgG,
                scratch.bgB);
            if (!captureKernelStage("prematte clean plate estimate", stream, waitForCompletion, error)) {
                return false;
            }

            if (!runGaussianBlur(scratch.bgR, scratch.scratch, scratch.gaussianWeights, width, height, max(1, params.prematteBlur), stream, waitForCompletion, error) ||
                !runGaussianBlur(scratch.bgG, scratch.scratch, scratch.gaussianWeights, width, height, max(1, params.prematteBlur), stream, waitForCompletion, error) ||
                !runGaussianBlur(scratch.bgB, scratch.scratch, scratch.gaussianWeights, width, height, max(1, params.prematteBlur), stream, waitForCompletion, error)) {
                return false;
            }

            packCleanPlateKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                width,
                height,
                scratch.bgR,
                scratch.bgG,
                scratch.bgB,
                scratch.cleanPlate);
            if (!captureKernelStage("prematte pack clean plate", stream, waitForCompletion, error)) {
                return false;
            }

            coreKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(params,
                                                                 frame.src,
                                                                 cleanScreen,
                                                                 frame.dst,
                                                                 frame.renderWindow.x1,
                                                                 frame.renderWindow.y1,
                                                                 width,
                                                                 height,
                                                                 scratch.rawAlpha,
                                                                 scratch.guide);
            if (!captureKernelStage("prematte rekey", stream, waitForCompletion, error)) {
                return false;
            }
        }

        if (params.viewMode == 2) {
            copyImageKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                cleanScreen,
                frame.dst,
                frame.renderWindow.x1,
                frame.renderWindow.y1,
                width,
                height);
            return captureKernelStage("diagnostic clean plate", stream, waitForCompletion, error);
        }
    } else if (params.viewMode == 2 && params.useScreenInput && frame.screen.data != nullptr) {
        copyImageKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            frame.screen,
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height);
        return captureKernelStage("diagnostic screen input", stream, waitForCompletion, error);
    }

    if (frame.garbageMatte.data != nullptr) {
        applyMatteKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            scratch.rawAlpha,
            frame.dst,
            frame.garbageMatte,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height,
            0);
        if (!captureKernelStage("garbage matte", stream, waitForCompletion, error)) {
            return false;
        }
    }
    if (frame.occlusionMatte.data != nullptr) {
        applyMatteKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            scratch.rawAlpha,
            frame.dst,
            frame.occlusionMatte,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height,
            1);
        if (!captureKernelStage("occlusion matte", stream, waitForCompletion, error)) {
            return false;
        }
    }

    if (params.viewMode == 1) {
        writeAlphaDiagnosticKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            scratch.rawAlpha,
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height);
        return captureKernelStage("diagnostic raw matte", stream, waitForCompletion, error);
    }

    if (doGF) {
        copyBufferKernel<<<flatBlocks, flatThreads, 0, stream>>>(pixelCount, scratch.rawAlpha, scratch.savedRawAlpha);
        if (!captureKernelStage("save raw alpha", stream, waitForCompletion, error)) {
            return false;
        }

        const int numIter = std::max(1, std::min(params.refineIterations, 5));
        if (doRgbGF) {
            for (int iter = 0; iter < numIter; ++iter) {
                rgbComputeProductsKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                    frame.src,
                    frame.renderWindow.x1,
                    frame.renderWindow.y1,
                    width,
                    height,
                    scratch.rawAlpha,
                    scratch.rgbMeanIr,
                    scratch.rgbMeanIg,
                    scratch.rgbMeanIb,
                    scratch.rgbMeanP,
                    scratch.rgbIrIr,
                    scratch.rgbIrIg,
                    scratch.rgbIrIb,
                    scratch.rgbIgIg,
                    scratch.rgbIgIb,
                    scratch.rgbIbIb,
                    scratch.rgbIrP,
                    scratch.rgbIgP,
                    scratch.rgbIbP);
                if (!captureKernelStage("rgb guided products", stream, waitForCompletion, error)) {
                    return false;
                }

                if (!runGaussianBlur(scratch.rgbMeanIr, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanIg, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanIb, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIrIr, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIrIg, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIrIb, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIgIg, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIgIb, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIbIb, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIrP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIgP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbIbP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error)) {
                    return false;
                }

                rgbGuidedCoeffKernel<<<flatBlocks, flatThreads, 0, stream>>>(
                    pixelCount,
                    params.guidedEpsilon,
                    scratch.rgbMeanIr,
                    scratch.rgbMeanIg,
                    scratch.rgbMeanIb,
                    scratch.rgbMeanP,
                    scratch.rgbIrIr,
                    scratch.rgbIrIg,
                    scratch.rgbIrIb,
                    scratch.rgbIgIg,
                    scratch.rgbIgIb,
                    scratch.rgbIbIb,
                    scratch.rgbIrP,
                    scratch.rgbIgP,
                    scratch.rgbIbP,
                    scratch.rgbMeanAr,
                    scratch.rgbMeanAg,
                    scratch.rgbMeanAb,
                    scratch.rgbMeanB);
                if (!captureKernelStage("rgb guided coefficients", stream, waitForCompletion, error)) {
                    return false;
                }

                if (!runGaussianBlur(scratch.rgbMeanAr, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanAg, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanAb, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.rgbMeanB, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error)) {
                    return false;
                }

                if (iter < numIter - 1) {
                    rgbGuidedEvalKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                        frame.src,
                        frame.renderWindow.x1,
                        frame.renderWindow.y1,
                        width,
                        height,
                        scratch.rgbMeanAr,
                        scratch.rgbMeanAg,
                        scratch.rgbMeanAb,
                        scratch.rgbMeanB,
                        scratch.rawAlpha);
                    if (!captureKernelStage("rgb guided eval", stream, waitForCompletion, error)) {
                        return false;
                    }
                }
            }

            rgbGuidedApplyKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                params,
                frame.src,
                frame.dst,
                frame.renderWindow.x1,
                frame.renderWindow.y1,
                width,
                height,
                scratch.savedRawAlpha,
                scratch.rgbMeanAr,
                scratch.rgbMeanAg,
                scratch.rgbMeanAb,
                scratch.rgbMeanB);
            if (!captureKernelStage("rgb guided apply", stream, waitForCompletion, error)) {
                return false;
            }
        } else {
            for (int iter = 0; iter < numIter; ++iter) {
                if (iter > 0) {
                    refineGuideKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                        width,
                        height,
                        frame.src,
                        frame.renderWindow.x1,
                        frame.renderWindow.y1,
                        params.edgeProtect,
                        scratch.rawAlpha,
                        scratch.guide);
                    if (!captureKernelStage("refine guide", stream, waitForCompletion, error)) {
                        return false;
                    }
                }

                computeProductsKernel<<<flatBlocks, flatThreads, 0, stream>>>(
                    pixelCount,
                    scratch.rawAlpha,
                    scratch.guide,
                    scratch.meanI,
                    scratch.meanP,
                    scratch.meanIp,
                    scratch.meanII);
                if (!captureKernelStage("guided products", stream, waitForCompletion, error)) {
                    return false;
                }

                if (!runGaussianBlur(scratch.meanI, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.meanP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.meanIp, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.meanII, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error)) {
                    return false;
                }

                guidedCoeffKernel<<<flatBlocks, flatThreads, 0, stream>>>(
                    pixelCount,
                    params.guidedEpsilon,
                    scratch.meanI,
                    scratch.meanP,
                    scratch.meanIp,
                    scratch.meanII);
                if (!captureKernelStage("guided coefficients", stream, waitForCompletion, error)) {
                    return false;
                }

                if (!runGaussianBlur(scratch.meanI, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error) ||
                    !runGaussianBlur(scratch.meanP, scratch.scratch, scratch.gaussianWeights, width, height, params.guidedRadius, stream, waitForCompletion, error)) {
                    return false;
                }

                if (iter < numIter - 1) {
                    guidedEvalKernel<<<flatBlocks, flatThreads, 0, stream>>>(
                        pixelCount,
                        scratch.guide,
                        scratch.meanI,
                        scratch.meanP,
                        scratch.rawAlpha);
                    if (!captureKernelStage("guided eval", stream, waitForCompletion, error)) {
                        return false;
                    }
                }
            }

            guidedApplyKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(params,
                                                                        frame.dst,
                                                                        frame.renderWindow.x1,
                                                                        frame.renderWindow.y1,
                                                                        width,
                                                                        height,
                                                                        scratch.savedRawAlpha,
                                                                        scratch.guide,
                                                                        scratch.meanI,
                                                                        scratch.meanP);
            if (!captureKernelStage("guided apply", stream, waitForCompletion, error)) {
                return false;
            }
        }
    } else if (params.premultiply) {
        premultiplyKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height);
        if (!captureKernelStage("premultiply", stream, waitForCompletion, error)) {
            return false;
        }
    }

    if (params.viewMode == 3) {
        extractOutputAlphaDiagnosticKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height);
        return captureKernelStage("diagnostic refined matte", stream, waitForCompletion, error);
    }

    if (params.edgeColorCorrect > 0.0f) {
        edgeColorCorrectKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            params,
            frame.src,
            frame.screen,
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height);
        if (!captureKernelStage("edge color correct", stream, waitForCompletion, error)) {
            return false;
        }
    }

    if (params.viewMode == 4) {
        return true;
    }

    if (needBgBlur) {
        extractBackgroundChannelsKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            width,
            height,
            frame.background,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            scratch.bgR,
            scratch.bgG,
            scratch.bgB);
        if (!captureKernelStage("background extract", stream, waitForCompletion, error)) {
            return false;
        }

        const int blurRadius = std::max(1, params.bgWrapBlur);
        if (!scratch.ensureGaussianWeights(buildGaussianWeights(blurRadius), blurRadius, error)) {
            return false;
        }
        if (!runGaussianBlur(scratch.bgR, scratch.scratch, scratch.gaussianWeights, width, height, blurRadius, stream, waitForCompletion, error) ||
            !runGaussianBlur(scratch.bgG, scratch.scratch, scratch.gaussianWeights, width, height, blurRadius, stream, waitForCompletion, error) ||
            !runGaussianBlur(scratch.bgB, scratch.scratch, scratch.gaussianWeights, width, height, blurRadius, stream, waitForCompletion, error)) {
            return false;
        }

        if (params.viewMode == 5) {
            packRgbKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                frame.dst,
                frame.renderWindow.x1,
                frame.renderWindow.y1,
                width,
                height,
                scratch.bgR,
                scratch.bgG,
                scratch.bgB);
            return captureKernelStage("diagnostic blurred background", stream, waitForCompletion, error);
        }

        if (doBgWrap) {
            bgWrapKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
                frame.dst,
                frame.renderWindow.x1,
                frame.renderWindow.y1,
                width,
                height,
                params.bgWrapAmount,
                scratch.bgR,
                scratch.bgG,
                scratch.bgB);
            if (!captureKernelStage("background wrap", stream, waitForCompletion, error)) {
                return false;
            }
        }
    }

    if (doAdditive) {
        additiveKeyKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(
            params,
            frame.src,
            frame.screen,
            frame.dst,
            frame.renderWindow.x1,
            frame.renderWindow.y1,
            width,
            height,
            (needBgBlur ? scratch.bgR : nullptr),
            (needBgBlur ? scratch.bgG : nullptr),
            (needBgBlur ? scratch.bgB : nullptr));
        if (!captureKernelStage("additive key", stream, waitForCompletion, error)) {
            return false;
        }
    }

    if (!waitForCompletion && !scratch.markInFlight(stream)) {
        // If we cannot record the fence event, we give up some performance and force the stream
        // to finish now so those scratch buffers are still safe to reuse on the next frame.
        const cudaError_t syncError = cudaStreamSynchronize(stream);
        if (syncError != cudaSuccess) {
            error = std::string("Host CUDA fallback sync failed after fence creation error: ") + cudaGetErrorString(syncError);
            return false;
        }
    }

    return true;
}

} // namespace

// New: OFX host-CUDA zero-copy execution.
//
// Use host-provided device pointers directly for best playback performance on hosts that support
// CUDA render and CUDA streams.
//
// "Zero-copy" here does not mean "no extra GPU memory exists anywhere". It means Source/Screen/
// Output stay on device. the guided filter still needs temporary GPU working storage.
bool renderCudaHost(const IBKeyerParams& params,
                    const DeviceRenderFrame& frame,
                    void* hostCudaStreamOpaque,
                    std::string& error)
{
    if (hostCudaStreamOpaque == nullptr) {
        error = "Host CUDA requires the host to provide a CUDA stream.";
        return false;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(hostCudaStreamOpaque);

    // In host-CUDA mode the host owns the destination image lifetime and synchronization contract.
    // Returning without a device-wide sync is what makes this path fast enough to be worth using.
    const bool waitForCompletion = hostCudaForceSyncEnabled();
    return renderCudaFrame(params, frame, stream, waitForCompletion, error);
}

////////////////////////////////////////////////////////////////////////////////
// STAGED CUDA FALLBACK
////////////////////////////////////////////////////////////////////////////////


// This is the compatibility path for hosts that do not support OFX host CUDA interop. It is slower
// than zero-copy, but it avoids regressing existing Windows/Linux functionality while keeping the
// actual CUDA math shared with the new fast path.
bool renderCudaInternal(const IBKeyerParams& params, const PackedFrame& frame, std::string& error)
{
    if (frame.width <= 0 || frame.height <= 0 || frame.srcRgba == nullptr || frame.dstRgba == nullptr) {
        error = "Invalid packed frame passed to CUDA.";
        return false;
    }

    const int pixelCount = frame.width * frame.height;
    const size_t rgbaBytes = static_cast<size_t>(pixelCount) * 4u * sizeof(float);

    float* dSrc = nullptr;
    float* dScreen = nullptr;
    float* dBackground = nullptr;
    float* dGarbageMatte = nullptr;
    float* dOcclusionMatte = nullptr;
    float* dDst = nullptr;
    DeviceRenderFrame deviceFrame;

    if (cudaMalloc(&dSrc, rgbaBytes) != cudaSuccess ||
        cudaMalloc(&dDst, rgbaBytes) != cudaSuccess ||
        (frame.screenRgba != nullptr && cudaMalloc(&dScreen, rgbaBytes) != cudaSuccess) ||
        (frame.backgroundRgba != nullptr && cudaMalloc(&dBackground, rgbaBytes) != cudaSuccess) ||
        (frame.garbageMatteRgba != nullptr && cudaMalloc(&dGarbageMatte, rgbaBytes) != cudaSuccess) ||
        (frame.occlusionMatteRgba != nullptr && cudaMalloc(&dOcclusionMatte, rgbaBytes) != cudaSuccess)) {
        error = "cudaMalloc failed for the staged CUDA path.";
        goto cleanup;
    }

    if (cudaMemcpy(dSrc, frame.srcRgba, rgbaBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "cudaMemcpy(src) failed for the staged CUDA path.";
        goto cleanup;
    }
    if (frame.screenRgba != nullptr &&
        cudaMemcpy(dScreen, frame.screenRgba, rgbaBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "cudaMemcpy(screen) failed for the staged CUDA path.";
        goto cleanup;
    }
    if (frame.backgroundRgba != nullptr &&
        cudaMemcpy(dBackground, frame.backgroundRgba, rgbaBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "cudaMemcpy(background) failed for the staged CUDA path.";
        goto cleanup;
    }
    if (frame.garbageMatteRgba != nullptr &&
        cudaMemcpy(dGarbageMatte, frame.garbageMatteRgba, rgbaBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "cudaMemcpy(garbage matte) failed for the staged CUDA path.";
        goto cleanup;
    }
    if (frame.occlusionMatteRgba != nullptr &&
        cudaMemcpy(dOcclusionMatte, frame.occlusionMatteRgba, rgbaBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "cudaMemcpy(occlusion matte) failed for the staged CUDA path.";
        goto cleanup;
    }

    deviceFrame.src.data = dSrc;
    deviceFrame.src.rowBytes = static_cast<size_t>(frame.width) * 4u * sizeof(float);
    deviceFrame.src.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.src.components = 4;
    deviceFrame.screen.data = dScreen;
    deviceFrame.screen.rowBytes = (dScreen != nullptr) ? static_cast<size_t>(frame.width) * 4u * sizeof(float) : 0u;
    deviceFrame.screen.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.screen.components = (dScreen != nullptr) ? 4 : 0;
    deviceFrame.background.data = dBackground;
    deviceFrame.background.rowBytes = (dBackground != nullptr) ? static_cast<size_t>(frame.width) * 4u * sizeof(float) : 0u;
    deviceFrame.background.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.background.components = (dBackground != nullptr) ? 4 : 0;
    deviceFrame.garbageMatte.data = dGarbageMatte;
    deviceFrame.garbageMatte.rowBytes = (dGarbageMatte != nullptr) ? static_cast<size_t>(frame.width) * 4u * sizeof(float) : 0u;
    deviceFrame.garbageMatte.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.garbageMatte.components = (dGarbageMatte != nullptr) ? 4 : 0;
    deviceFrame.occlusionMatte.data = dOcclusionMatte;
    deviceFrame.occlusionMatte.rowBytes = (dOcclusionMatte != nullptr) ? static_cast<size_t>(frame.width) * 4u * sizeof(float) : 0u;
    deviceFrame.occlusionMatte.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.occlusionMatte.components = (dOcclusionMatte != nullptr) ? 4 : 0;
    deviceFrame.dst.data = dDst;
    deviceFrame.dst.rowBytes = static_cast<size_t>(frame.width) * 4u * sizeof(float);
    deviceFrame.dst.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.dst.components = 4;
    deviceFrame.renderWindow = {0, 0, frame.width, frame.height};

    // The fallback path still stages through packed buffers because the host did not give us
    // usable CUDA images. We keep it around so the plugin can run on more hosts without forcing
    // the zero-copy contract onto environments that do not support it.
    if (!renderCudaFrame(params, deviceFrame, nullptr, true, error)) {
        goto cleanup;
    }

    if (cudaMemcpy(frame.dstRgba, dDst, rgbaBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        error = "cudaMemcpy(dst) failed for the staged CUDA path.";
        goto cleanup;
    }

cleanup:
    cudaFree(dDst);
    cudaFree(dOcclusionMatte);
    cudaFree(dGarbageMatte);
    cudaFree(dBackground);
    cudaFree(dScreen);
    cudaFree(dSrc);
    return error.empty();
}

} // namespace IBKeyerCore

#else

namespace IBKeyerCore {

bool renderCudaHost(const IBKeyerParams&, const DeviceRenderFrame&, void*, std::string& error)
{
    error = "CUDA support was not compiled for this build.";
    return false;
}

bool renderCudaInternal(const IBKeyerParams&, const PackedFrame&, std::string& error)
{
    error = "CUDA support was not compiled for this build.";
    return false;
}

} // namespace IBKeyerCore

#endif
