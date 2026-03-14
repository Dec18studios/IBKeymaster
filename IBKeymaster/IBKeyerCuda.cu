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
    float* guide = nullptr;
    float* meanI = nullptr;
    float* meanP = nullptr;
    float* meanIp = nullptr;
    float* meanII = nullptr;
    float* scratch = nullptr;
    int pixelCapacity = 0;

    float* gaussianWeights = nullptr;
    int weightRadius = -1;
    size_t weightCount = 0;
    cudaEvent_t inFlightEvent = nullptr;
    cudaStream_t lastStream = nullptr;

    void release()
    {
        cudaFree(gaussianWeights);
        cudaFree(scratch);
        cudaFree(meanII);
        cudaFree(meanIp);
        cudaFree(meanP);
        cudaFree(meanI);
        cudaFree(guide);
        cudaFree(rawAlpha);
        if (inFlightEvent != nullptr) {
            cudaEventDestroy(inFlightEvent);
        }

        gaussianWeights = nullptr;
        scratch = nullptr;
        meanII = nullptr;
        meanIp = nullptr;
        meanP = nullptr;
        meanI = nullptr;
        guide = nullptr;
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

        cudaFree(scratch);
        cudaFree(meanII);
        cudaFree(meanIp);
        cudaFree(meanP);
        cudaFree(meanI);
        cudaFree(guide);
        cudaFree(rawAlpha);

        scratch = nullptr;
        meanII = nullptr;
        meanIp = nullptr;
        meanP = nullptr;
        meanI = nullptr;
        guide = nullptr;
        rawAlpha = nullptr;

        const size_t channelBytes = static_cast<size_t>(pixelCount) * sizeof(float);
        if (cudaMalloc(&rawAlpha, channelBytes) != cudaSuccess ||
            cudaMalloc(&guide, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanI, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanP, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanIp, channelBytes) != cudaSuccess ||
            cudaMalloc(&meanII, channelBytes) != cudaSuccess ||
            cudaMalloc(&scratch, channelBytes) != cudaSuccess) {
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
    const float respillMul = despillScreen * normalized > 0.0f ? despillScreen * normalized : 0.0f;
    storeRgba(dst,
              imageX,
              imageY,
              ssR + respillMul * params.respillR,
              ssG + respillMul * params.respillG,
              ssB + respillMul * params.respillB,
              alpha);

    rawAlpha[pixelIndex] = alpha;
    guide[pixelIndex] = luminance(srcR, srcG, srcB);
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

    if (doGF) {
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

        // This is the part zero-copy does not remove: the guided filter still needs temporary
        // working buffers on the GPU. What zero-copy changes is that Source/Screen/Output stay
        // on the host-owned device images instead of bouncing through CPU memory first.
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

        guidedApplyKernel<<<pixelBlocks, pixelThreads, 0, stream>>>(params,
                                                                    frame.dst,
                                                                    frame.renderWindow.x1,
                                                                    frame.renderWindow.y1,
                                                                    width,
                                                                    height,
                                                                    scratch.rawAlpha,
                                                                    scratch.guide,
                                                                    scratch.meanI,
                                                                    scratch.meanP);
        if (!captureKernelStage("guided apply", stream, waitForCompletion, error)) {
            return false;
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
    float* dDst = nullptr;
    DeviceRenderFrame deviceFrame;

    if (cudaMalloc(&dSrc, rgbaBytes) != cudaSuccess ||
        cudaMalloc(&dDst, rgbaBytes) != cudaSuccess ||
        (frame.screenRgba != nullptr && cudaMalloc(&dScreen, rgbaBytes) != cudaSuccess)) {
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

    deviceFrame.src.data = dSrc;
    deviceFrame.src.rowBytes = static_cast<size_t>(frame.width) * 4u * sizeof(float);
    deviceFrame.src.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.src.components = 4;
    deviceFrame.screen.data = dScreen;
    deviceFrame.screen.rowBytes = (dScreen != nullptr) ? static_cast<size_t>(frame.width) * 4u * sizeof(float) : 0u;
    deviceFrame.screen.bounds = {0, 0, frame.width, frame.height};
    deviceFrame.screen.components = (dScreen != nullptr) ? 4 : 0;
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
