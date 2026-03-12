#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
#include <cmath>

/*
 * IBKeymaster v2.1 — Guided-Filter Enhanced Metal Kernel
 *
 * Multi-pass architecture:
 *   Pass 1: Core IBK key extraction (color-difference → raw alpha + despilled FG)
 *   Pass 2+: Guided Filter matte refinement (iterable, true Gaussian blur)
 *       - Compute products I*p, I*I
 *       - Gaussian blur (separable H+V, pre-computed weights)
 *       - Coefficient computation (a, b from local statistics)
 *       - Blur coefficients → mean_a, mean_b
 *       - Evaluate / apply: refined_alpha = mean_a * I + mean_b
 *       - (optional) Iterate: use refined alpha to compute better FG estimate,
 *         update guide, re-run guided filter for progressively cleaner matte
 *   Final: Mix with raw alpha + premultiply (if requested)
 */

const char* kernelSource = R"(
#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════

inline void reorderChannels(float r, float g, float b, int sc,
                            thread float& c0, thread float& c1, thread float& c2)
{
    if (sc == 0)      { c0 = r; c1 = g; c2 = b; }
    else if (sc == 1) { c0 = g; c1 = r; c2 = b; }
    else              { c0 = b; c1 = r; c2 = g; }
}

inline float despillVal(float r, float g, float b, int sc, float bias, float lim)
{
    float c0, c1, c2;
    reorderChannels(r, g, b, sc, c0, c1, c2);
    return c0 - (c1 * bias + c2 * (1.0f - bias)) * lim;
}

inline float safeDivide(float a, float b)
{
    return (abs(b) > 1e-8f) ? a / b : 0.0f;
}

inline float luminance(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// ════════════════════════════════════════════════════════════════════════
//  Core IBK Key Extraction
//  Outputs: RGBA (despilled FG + raw alpha), raw alpha (1ch), guide (1ch)
// ════════════════════════════════════════════════════════════════════════

kernel void IBKeymasterCoreKernel(
    const device float* p_Input       [[buffer(0)]],
    const device float* p_Screen      [[buffer(1)]],
    device float*       p_Output      [[buffer(2)]],
    device float*       p_RawAlpha    [[buffer(3)]],
    device float*       p_Guide       [[buffer(4)]],
    constant int&       p_Width       [[buffer(10)]],
    constant int&       p_Height      [[buffer(11)]],
    constant int&       p_ScreenColor [[buffer(12)]],
    constant int&       p_UseScreenInput [[buffer(13)]],
    constant float&     p_PickR       [[buffer(14)]],
    constant float&     p_PickG       [[buffer(15)]],
    constant float&     p_PickB       [[buffer(16)]],
    constant float&     p_Bias        [[buffer(17)]],
    constant float&     p_Limit       [[buffer(18)]],
    constant float&     p_RespillR    [[buffer(19)]],
    constant float&     p_RespillG    [[buffer(20)]],
    constant float&     p_RespillB    [[buffer(21)]],
    constant int&       p_NearGreyExtract [[buffer(22)]],
    constant float&     p_NearGreyAmount  [[buffer(23)]],
    constant float&     p_NearGreySoftness [[buffer(28)]],
    constant float&     p_BlackClip   [[buffer(24)]],
    constant float&     p_WhiteClip   [[buffer(25)]],
    constant float&     p_EdgeProtect [[buffer(26)]],
    constant float&     p_MatteGamma  [[buffer(27)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;

    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float srcR = p_Input[idx4 + 0];
    float srcG = p_Input[idx4 + 1];
    float srcB = p_Input[idx4 + 2];

    float scrR, scrG, scrB;
    if (p_UseScreenInput != 0) {
        scrR = p_Screen[idx4 + 0];
        scrG = p_Screen[idx4 + 1];
        scrB = p_Screen[idx4 + 2];
    } else {
        scrR = p_PickR;
        scrG = p_PickG;
        scrB = p_PickB;
    }

    int sc = p_ScreenColor;
    float bias = p_Bias;
    float lim  = p_Limit;

    // 1. Despill source and screen
    float dSrc = despillVal(srcR, srcG, srcB, sc, bias, lim);
    float dScr = despillVal(scrR, scrG, scrB, sc, bias, lim);

    // 2. Normalise
    float norm = safeDivide(dSrc, dScr);

    // 3. Screen subtraction
    float spillMul = max(0.0f, norm);
    float ssR = srcR - spillMul * scrR;
    float ssG = srcG - spillMul * scrG;
    float ssB = srcB - spillMul * scrB;

    // 4. Initial alpha
    float alpha = clamp(1.0f - norm, 0.0f, 1.0f);

    // 5. Near Grey Extraction
    if (p_NearGreyExtract != 0 && p_NearGreyAmount > 0.0f) {
        float divR = safeDivide(ssR, srcR);
        float divG = safeDivide(ssG, srcG);
        float divB = safeDivide(ssB, srcB);
        float c0, c1, c2;
        reorderChannels(divR, divG, divB, sc, c0, c1, c2);
        float mx = max(c0, max(c1, c2));
        float comp = (mx == c1) ? c1 : c2;
        float ngeA = clamp(c0 * (1.0f - p_NearGreySoftness) + comp * p_NearGreySoftness, 0.0f, 1.0f);
        // Strength-blended screen composite:
        // full screen = ngeA + alpha - ngeA * alpha = alpha + ngeA*(1-alpha)
        // blend by amount: alpha + amount * ngeA * (1 - alpha)
        alpha = alpha + p_NearGreyAmount * ngeA * (1.0f - alpha);
    }

    // 6. Black/White clip (levels on raw alpha)
    float lo = p_BlackClip;
    float hi = p_WhiteClip;
    if (hi > lo + 1e-6f) {
        alpha = clamp((alpha - lo) / (hi - lo), 0.0f, 1.0f);
    }

    // 6b. Matte Gamma — shape alpha falloff (helps motion blur)
    if (p_MatteGamma != 1.0f && alpha > 0.0f && alpha < 1.0f) {
        alpha = pow(alpha, p_MatteGamma);
    }

    // 7. Respill
    float rspMul = max(0.0f, dScr * norm);
    float outR = ssR + rspMul * p_RespillR;
    float outG = ssG + rspMul * p_RespillG;
    float outB = ssB + rspMul * p_RespillB;

    // Write despilled FG + raw alpha to output
    p_Output[idx4 + 0] = outR;
    p_Output[idx4 + 1] = outG;
    p_Output[idx4 + 2] = outB;
    p_Output[idx4 + 3] = alpha;

    // Write single-channel temps for guided filter
    p_RawAlpha[idx1] = alpha;
    float lum = luminance(srcR, srcG, srcB);
    p_Guide[idx1] = lum * (1.0f - p_EdgeProtect) + alpha * p_EdgeProtect;
}

// ════════════════════════════════════════════════════════════════════════
//  Gaussian Blur — Horizontal (single channel, pre-computed weights)
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurH(
    const device float* p_Src     [[buffer(3)]],
    device float*       p_Dst     [[buffer(4)]],
    const device float* p_Weights [[buffer(5)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Radius  [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;

    int x = (int)id.x;
    int y = (int)id.y;
    int r = p_Radius;
    int w = p_Width;

    float sum = 0.0f;
    for (int dx = -r; dx <= r; dx++) {
        int sx = clamp(x + dx, 0, w - 1);
        sum += p_Src[y * w + sx] * p_Weights[dx + r];
    }
    p_Dst[y * w + x] = sum;
}

// ════════════════════════════════════════════════════════════════════════
//  Gaussian Blur — Vertical (single channel, pre-computed weights)
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurV(
    const device float* p_Src     [[buffer(3)]],
    device float*       p_Dst     [[buffer(4)]],
    const device float* p_Weights [[buffer(5)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Radius  [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;

    int x = (int)id.x;
    int y = (int)id.y;
    int r = p_Radius;
    int h = p_Height;
    int w = p_Width;

    float sum = 0.0f;
    for (int dy = -r; dy <= r; dy++) {
        int sy = clamp(y + dy, 0, h - 1);
        sum += p_Src[sy * w + x] * p_Weights[dy + r];
    }
    p_Dst[y * w + x] = sum;
}

// ════════════════════════════════════════════════════════════════════════
//  Compute products: I*p and I*I, and copy I for blurring
// ════════════════════════════════════════════════════════════════════════

kernel void ComputeProducts(
    const device float* p_RawAlpha [[buffer(3)]],
    device float*       p_CopyI    [[buffer(4)]],
    const device float* p_Guide    [[buffer(5)]],
    device float*       p_Ip       [[buffer(6)]],
    device float*       p_II       [[buffer(7)]],
    constant int&       p_Width    [[buffer(10)]],
    constant int&       p_Height   [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx = (int)(id.y * (uint)p_Width) + (int)id.x;

    float I = p_Guide[idx];
    float p = p_RawAlpha[idx];

    p_CopyI[idx] = I;
    p_Ip[idx] = I * p;
    p_II[idx] = I * I;
}

// ════════════════════════════════════════════════════════════════════════
//  Guided Filter — Coefficient computation
//  a = cov(I,p) / (var(I) + eps)
//  b = mean_p - a * mean_I
//  Overwrites: buf[3] → a, buf[4] → b
// ════════════════════════════════════════════════════════════════════════

kernel void GuidedFilterCoeff(
    device float*       p_MeanI   [[buffer(3)]],
    device float*       p_MeanP   [[buffer(4)]],
    const device float* p_MeanIp  [[buffer(5)]],
    const device float* p_MeanII  [[buffer(6)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant float&     p_Epsilon [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx = (int)(id.y * (uint)p_Width) + (int)id.x;

    float mI  = p_MeanI[idx];
    float mP  = p_MeanP[idx];
    float mIp = p_MeanIp[idx];
    float mII = p_MeanII[idx];

    float varI  = mII - mI * mI;
    float covIp = mIp - mI * mP;

    float a = covIp / (varI + p_Epsilon);
    float b = mP - a * mI;

    p_MeanI[idx] = a;
    p_MeanP[idx] = b;
}

// ════════════════════════════════════════════════════════════════════════
//  Refine Guide — for iterative refinement (iteration 2+)
//  Uses refined alpha from previous iteration to compute better FG
//  estimate, then builds an improved guide signal from that.
// ════════════════════════════════════════════════════════════════════════

kernel void RefineGuideKernel(
    const device float* p_Input       [[buffer(0)]],
    const device float* p_Alpha       [[buffer(3)]],
    device float*       p_Guide       [[buffer(5)]],
    constant int&       p_Width       [[buffer(10)]],
    constant int&       p_Height      [[buffer(11)]],
    constant float&     p_EdgeProtect [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;

    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float alpha = p_Alpha[idx1];
    float srcR = p_Input[idx4 + 0];
    float srcG = p_Input[idx4 + 1];
    float srcB = p_Input[idx4 + 2];

    // FG-weighted luminance: premultiply by alpha for cleaner edge separation
    float fgLum = luminance(srcR * alpha, srcG * alpha, srcB * alpha);

    // Blend guide: FG luminance → alpha (self-guided) by edge protection
    p_Guide[idx1] = fgLum * (1.0f - p_EdgeProtect) + alpha * p_EdgeProtect;
}

// ════════════════════════════════════════════════════════════════════════
//  Guided Filter — Evaluate (intermediate iteration)
//  Writes refined alpha to a 1-channel buffer (no mix, no premultiply)
// ════════════════════════════════════════════════════════════════════════

kernel void GuidedFilterEval(
    const device float* p_MeanA  [[buffer(3)]],
    const device float* p_MeanB  [[buffer(4)]],
    const device float* p_Guide  [[buffer(5)]],
    device float*       p_Out    [[buffer(6)]],
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx = (int)(id.y * (uint)p_Width) + (int)id.x;
    p_Out[idx] = clamp(p_MeanA[idx] * p_Guide[idx] + p_MeanB[idx], 0.0f, 1.0f);
}

// ════════════════════════════════════════════════════════════════════════
//  Guided Filter — Final Apply
//  refined_alpha = mix(rawAlpha, guided_alpha, gfMix)
//  Then optionally premultiply RGB.
// ════════════════════════════════════════════════════════════════════════

kernel void GuidedFilterApply(
    device float*       p_Output    [[buffer(2)]],
    const device float* p_MeanA     [[buffer(3)]],
    const device float* p_MeanB     [[buffer(4)]],
    const device float* p_Guide     [[buffer(5)]],
    const device float* p_RawAlpha  [[buffer(6)]],
    constant int&       p_Width     [[buffer(10)]],
    constant int&       p_Height    [[buffer(11)]],
    constant int&       p_Premultiply [[buffer(12)]],
    constant float&     p_GFMix     [[buffer(13)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;

    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float rawAlpha = p_RawAlpha[idx1];
    float guidedAlpha = clamp(p_MeanA[idx1] * p_Guide[idx1] + p_MeanB[idx1], 0.0f, 1.0f);
    float alpha = mix(rawAlpha, guidedAlpha, p_GFMix);

    float outR = p_Output[idx4 + 0];
    float outG = p_Output[idx4 + 1];
    float outB = p_Output[idx4 + 2];

    if (p_Premultiply != 0) {
        outR *= alpha;
        outG *= alpha;
        outB *= alpha;
    }

    p_Output[idx4 + 0] = outR;
    p_Output[idx4 + 1] = outG;
    p_Output[idx4 + 2] = outB;
    p_Output[idx4 + 3] = alpha;
}

// ════════════════════════════════════════════════════════════════════════
//  Simple premultiply (when guided filter is off but premultiply is on)
// ════════════════════════════════════════════════════════════════════════

kernel void PremultiplyKernel(
    device float*   p_Output  [[buffer(2)]],
    constant int&   p_Width   [[buffer(10)]],
    constant int&   p_Height  [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;

    float a = p_Output[idx4 + 3];
    p_Output[idx4 + 0] *= a;
    p_Output[idx4 + 1] *= a;
    p_Output[idx4 + 2] *= a;
}

// ════════════════════════════════════════════════════════════════════════
//  Copy single-channel buffer: buf[3] → buf[4]
// ════════════════════════════════════════════════════════════════════════

kernel void CopyBuffer(
    const device float* p_Src    [[buffer(3)]],
    device float*       p_Dst    [[buffer(4)]],
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx = (int)(id.y * (uint)p_Width) + (int)id.x;
    p_Dst[idx] = p_Src[idx];
}

// ════════════════════════════════════════════════════════════════════════
//  Extract one channel from RGBA buffer into 1ch buffer
//  buf[0] = RGBA source, buf[3] = 1ch destination
//  buffer(12) = channel index (0=R, 1=G, 2=B)
// ════════════════════════════════════════════════════════════════════════

kernel void ExtractChannel(
    const device float* p_RGBA    [[buffer(0)]],
    device float*       p_Out     [[buffer(3)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Chan    [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;
    p_Out[idx1] = p_RGBA[idx4 + p_Chan];
}

// ════════════════════════════════════════════════════════════════════════
//  Background Wrap: blend blurred BG into FG edges
//  buf[2] = output RGBA (read+write)
//  buf[3] = blurred BG red (1ch)
//  buf[4] = blurred BG green (1ch)
//  buf[5] = blurred BG blue (1ch)
//  buffer(12) = amount (float)
// ════════════════════════════════════════════════════════════════════════

kernel void BgWrapKernel(
    device float*       p_Output  [[buffer(2)]],
    const device float* p_BgR     [[buffer(3)]],
    const device float* p_BgG     [[buffer(4)]],
    const device float* p_BgB     [[buffer(5)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant float&     p_Amount  [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;

    float alpha = p_Output[idx4 + 3];
    float w = alpha * (1.0f - alpha) * 4.0f * p_Amount;

    p_Output[idx4 + 0] += p_BgR[idx1] * w;
    p_Output[idx4 + 1] += p_BgG[idx1] * w;
    p_Output[idx4 + 2] += p_BgB[idx1] * w;
}

// ════════════════════════════════════════════════════════════════════════
//  Edge Color Correction: re-estimate FG color at semi-transparent edges
//  using the matting equation: fg = (src - screen*(1-alpha)) / alpha
//  Corrects residual screen contamination the despill couldn't reach.
// ════════════════════════════════════════════════════════════════════════

kernel void EdgeColorCorrectKernel(
    const device float* p_Input    [[buffer(0)]],
    const device float* p_Screen   [[buffer(1)]],
    device float*       p_Output   [[buffer(2)]],
    constant int&       p_Width    [[buffer(10)]],
    constant int&       p_Height   [[buffer(11)]],
    constant int&       p_UseScreen [[buffer(12)]],
    constant float&     p_PickR    [[buffer(13)]],
    constant float&     p_PickG    [[buffer(14)]],
    constant float&     p_PickB    [[buffer(15)]],
    constant float&     p_Amount   [[buffer(16)]],
    constant int&       p_Premult  [[buffer(17)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;

    float alpha = p_Output[idx4 + 3];

    // Only correct semi-transparent edge pixels
    if (alpha < 0.005f || alpha > 0.995f) return;

    float srcR = p_Input[idx4 + 0];
    float srcG = p_Input[idx4 + 1];
    float srcB = p_Input[idx4 + 2];

    float scrR, scrG, scrB;
    if (p_UseScreen != 0) {
        scrR = p_Screen[idx4 + 0];
        scrG = p_Screen[idx4 + 1];
        scrB = p_Screen[idx4 + 2];
    } else {
        scrR = p_PickR; scrG = p_PickG; scrB = p_PickB;
    }

    // Matting equation: fg = (src - screen * (1-alpha)) / alpha
    float invA = 1.0f / alpha;
    float fgR = (srcR - scrR * (1.0f - alpha)) * invA;
    float fgG = (srcG - scrG * (1.0f - alpha)) * invA;
    float fgB = (srcB - scrB * (1.0f - alpha)) * invA;

    // Soft clamp to prevent extreme values at low alpha
    fgR = clamp(fgR, -0.5f, 2.0f);
    fgG = clamp(fgG, -0.5f, 2.0f);
    fgB = clamp(fgB, -0.5f, 2.0f);

    // Current output RGB
    float curR = p_Output[idx4 + 0];
    float curG = p_Output[idx4 + 1];
    float curB = p_Output[idx4 + 2];

    // Un-premultiply if needed for correct blending
    if (p_Premult != 0) {
        curR *= invA; curG *= invA; curB *= invA;
    }

    // Edge factor: bell curve peaking at alpha=0.5
    float ef = alpha * (1.0f - alpha) * 4.0f * p_Amount;

    float outR = curR + (fgR - curR) * ef;
    float outG = curG + (fgG - curG) * ef;
    float outB = curB + (fgB - curB) * ef;

    // Re-premultiply if needed
    if (p_Premult != 0) {
        outR *= alpha; outG *= alpha; outB *= alpha;
    }

    p_Output[idx4 + 0] = outR;
    p_Output[idx4 + 1] = outG;
    p_Output[idx4 + 2] = outB;
}
)";

// ═══════════════════════════════════════════════════════════════════════════════
//  Pipeline cache
// ═══════════════════════════════════════════════════════════════════════════════

struct PipelineSet {
    id<MTLComputePipelineState> coreKeyer;
    id<MTLComputePipelineState> gaussianBlurH;
    id<MTLComputePipelineState> gaussianBlurV;
    id<MTLComputePipelineState> computeProducts;
    id<MTLComputePipelineState> guidedCoeff;
    id<MTLComputePipelineState> refineGuide;
    id<MTLComputePipelineState> guidedEval;
    id<MTLComputePipelineState> guidedApply;
    id<MTLComputePipelineState> premultiply;
    id<MTLComputePipelineState> copyBuffer;
    id<MTLComputePipelineState> extractChannel;
    id<MTLComputePipelineState> bgWrap;
    id<MTLComputePipelineState> edgeColorCorrect;
};

struct QueueState {
    PipelineSet pipes;
    // Cached temp buffers (persist across frames — avoids alloc/free churn)
    //   tempA: raw alpha / p / mean_p / b / mean_b
    //   tempB: guide / I / mean_I / a / mean_a
    //   tempC: I*p / mean_Ip / eval scratch
    //   tempD: I*I / mean_II
    //   tempE: guide copy (preserved through blur passes)
    //   tempF: Gaussian blur scratch (H/V intermediate)
    //   tempG: saved raw alpha (for final mix against original)
    id<MTLBuffer> tempA = nil;
    id<MTLBuffer> tempB = nil;
    id<MTLBuffer> tempC = nil;
    id<MTLBuffer> tempD = nil;
    id<MTLBuffer> tempE = nil;
    id<MTLBuffer> tempF = nil;
    id<MTLBuffer> tempG = nil;
    size_t cachedChanBytes = 0;
};

std::mutex s_PipelineMutex;
typedef std::unordered_map<id<MTLCommandQueue>, QueueState> QueueStateMap;
QueueStateMap s_QueueStateMap;

static id<MTLComputePipelineState> makePipeline(id<MTLLibrary> lib, const char* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "IBKeymaster: kernel '%s' not found\n", name); return nil; }
    id<MTLComputePipelineState> ps = [lib.device newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!ps) fprintf(stderr, "IBKeymaster: pipeline '%s' failed: %s\n", name, err.localizedDescription.UTF8String);
    return ps;
}

static void dispatch2D(id<MTLComputeCommandEncoder> enc,
                       id<MTLComputePipelineState> ps, int w, int h)
{
    int exeW = (int)[ps threadExecutionWidth];
    int maxT = (int)[ps maxTotalThreadsPerThreadgroup];
    int grpH = maxT / exeW;
    if (grpH < 1) grpH = 1;
    MTLSize tgSize = MTLSizeMake(exeW, grpH, 1);
    MTLSize tgCount = MTLSizeMake((w + exeW - 1) / exeW, (h + grpH - 1) / grpH, 1);
    [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Gaussian blur helper: single separable H+V pass with pre-computed weights.
//  Blurs bufA in-place, uses bufScratch as intermediate.
// ═══════════════════════════════════════════════════════════════════════════════

static void gaussianBlur(id<MTLComputeCommandEncoder> enc,
                         const PipelineSet& ps,
                         id<MTLBuffer> bufA, id<MTLBuffer> bufScratch,
                         id<MTLBuffer> weightBuf,
                         int w, int h, int radius)
{
    // Horizontal: A → scratch
    [enc setComputePipelineState:ps.gaussianBlurH];
    [enc setBuffer:bufA       offset:0 atIndex:3];
    [enc setBuffer:bufScratch offset:0 atIndex:4];
    [enc setBuffer:weightBuf  offset:0 atIndex:5];
    [enc setBytes:&w      length:sizeof(int) atIndex:10];
    [enc setBytes:&h      length:sizeof(int) atIndex:11];
    [enc setBytes:&radius length:sizeof(int) atIndex:12];
    dispatch2D(enc, ps.gaussianBlurH, w, h);

    // Vertical: scratch → A
    [enc setComputePipelineState:ps.gaussianBlurV];
    [enc setBuffer:bufScratch offset:0 atIndex:3];
    [enc setBuffer:bufA       offset:0 atIndex:4];
    [enc setBuffer:weightBuf  offset:0 atIndex:5];
    [enc setBytes:&w      length:sizeof(int) atIndex:10];
    [enc setBytes:&h      length:sizeof(int) atIndex:11];
    [enc setBytes:&radius length:sizeof(int) atIndex:12];
    dispatch2D(enc, ps.gaussianBlurV, w, h);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main dispatch
// ═══════════════════════════════════════════════════════════════════════════════

void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                    int p_ScreenColor, int p_UseScreenInput,
                    float p_PickR, float p_PickG, float p_PickB,
                    float p_Bias, float p_Limit,
                    float p_RespillR, float p_RespillG, float p_RespillB,
                    int p_Premultiply, int p_NearGreyExtract,
                    float p_NearGreyAmount, float p_NearGreySoftness,
                    float p_BlackClip, float p_WhiteClip, float p_MatteGamma,
                    int p_GuidedFilterEnabled, int p_GuidedRadius, float p_GuidedEpsilon,
                    float p_GuidedMix, float p_EdgeProtect, int p_RefineIterations,
                    float p_EdgeColorCorrect,
                    int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount,
                    const float* p_Input, const float* p_Screen,
                    const float* p_Background, float* p_Output)
{
  @autoreleasepool {
    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(p_CmdQ);
    id<MTLDevice> device = queue.device;
    NSError* err = nil;

    // ── Pipeline + buffer cache ──
    std::unique_lock<std::mutex> lock(s_PipelineMutex);
    auto it = s_QueueStateMap.find(queue);
    if (it == s_QueueStateMap.end()) {
        MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        options.mathMode = MTLMathModeFast;
#else
        options.fastMathEnabled = YES;
#endif
        id<MTLLibrary> lib = [device newLibraryWithSource:@(kernelSource) options:options error:&err];
        [options release];
        if (!lib) {
            fprintf(stderr, "IBKeymaster: Metal compile failed: %s\n", err.localizedDescription.UTF8String);
            return;
        }
        QueueState qs;
        qs.pipes.coreKeyer       = makePipeline(lib, "IBKeymasterCoreKernel");
        qs.pipes.gaussianBlurH   = makePipeline(lib, "GaussianBlurH");
        qs.pipes.gaussianBlurV   = makePipeline(lib, "GaussianBlurV");
        qs.pipes.computeProducts = makePipeline(lib, "ComputeProducts");
        qs.pipes.guidedCoeff     = makePipeline(lib, "GuidedFilterCoeff");
        qs.pipes.refineGuide     = makePipeline(lib, "RefineGuideKernel");
        qs.pipes.guidedEval      = makePipeline(lib, "GuidedFilterEval");
        qs.pipes.guidedApply     = makePipeline(lib, "GuidedFilterApply");
        qs.pipes.premultiply     = makePipeline(lib, "PremultiplyKernel");
        qs.pipes.copyBuffer      = makePipeline(lib, "CopyBuffer");
        qs.pipes.extractChannel  = makePipeline(lib, "ExtractChannel");
        qs.pipes.bgWrap          = makePipeline(lib, "BgWrapKernel");
        qs.pipes.edgeColorCorrect = makePipeline(lib, "EdgeColorCorrectKernel");
        [lib release];
        s_QueueStateMap[queue] = qs;
        it = s_QueueStateMap.find(queue);
    }
    QueueState& state = it->second;
    PipelineSet& pipes = state.pipes;

    // ── Cached temp buffers (reused across frames) ──
    size_t chanBytes = (size_t)p_Width * (size_t)p_Height * sizeof(float);
    bool doGF = p_GuidedFilterEnabled && p_GuidedRadius > 0;
    bool doBgWrap = p_BgWrapEnabled && p_Background && p_BgWrapAmount > 0.0f;

    // Reallocate if resolution changed
    if (chanBytes != state.cachedChanBytes) {
        if (state.tempA) [state.tempA release];
        if (state.tempB) [state.tempB release];
        if (state.tempC) [state.tempC release];
        if (state.tempD) [state.tempD release];
        if (state.tempE) [state.tempE release];
        if (state.tempF) [state.tempF release];
        if (state.tempG) [state.tempG release];
        state.tempA = nil; state.tempB = nil; state.tempC = nil;
        state.tempD = nil; state.tempE = nil; state.tempF = nil;
        state.tempG = nil;
        state.cachedChanBytes = chanBytes;
    }

    // Allocate on demand
    if ((doGF || doBgWrap) && !state.tempC) {
        // Need buffers for GF and/or BG wrap
        if (!state.tempA) state.tempA = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        if (!state.tempB) state.tempB = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempC = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempD = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempE = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempF = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempG = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
    } else if (!doGF && !doBgWrap && !state.tempA) {
        // Need tempA and tempB even without GF (core kernel writes to them)
        state.tempA = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        state.tempB = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
    }

    id<MTLBuffer> tempA = state.tempA;
    id<MTLBuffer> tempB = state.tempB;
    id<MTLBuffer> tempC = state.tempC;
    id<MTLBuffer> tempD = state.tempD;
    id<MTLBuffer> tempE = state.tempE;
    id<MTLBuffer> tempF = state.tempF;
    id<MTLBuffer> tempG = state.tempG;
    lock.unlock();

    // ── Resolve Metal buffers ──
    id<MTLBuffer> srcBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> dstBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    id<MTLBuffer> scrBuf = nil;
    bool createdDummy = false;
    if (p_Screen && p_UseScreenInput) {
        scrBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Screen));
    } else {
        float dummy[4] = {0, 0, 0, 0};
        scrBuf = [device newBufferWithBytes:dummy length:sizeof(dummy)
                                    options:MTLResourceStorageModeShared];
        createdDummy = true;
    }

    // ── Pre-compute Gaussian weights (tiny shared buffer, freed per-frame) ──
    id<MTLBuffer> weightBuf = nil;
    if (doGF) {
        int r = p_GuidedRadius;
        int kernelSize = 2 * r + 1;
        float sigma = fmaxf(r / 3.0f, 0.5f);
        float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

        float* weights = (float*)alloca(kernelSize * sizeof(float));
        float wsum = 0.0f;
        for (int i = -r; i <= r; i++) {
            float w = expf(-(float)(i * i) * invTwoSigmaSq);
            weights[i + r] = w;
            wsum += w;
        }
        for (int i = 0; i < kernelSize; i++) weights[i] /= wsum;

        weightBuf = [device newBufferWithBytes:weights
                                        length:kernelSize * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PASS 1: Core keyer
    // ══════════════════════════════════════════════════════════════════════
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"IBKeymaster";
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:pipes.coreKeyer];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:scrBuf offset:0 atIndex:1];
    [enc setBuffer:dstBuf offset:0 atIndex:2];
    [enc setBuffer:tempA  offset:0 atIndex:3];   // raw alpha out
    [enc setBuffer:tempB  offset:0 atIndex:4];   // guide out
    [enc setBytes:&p_Width           length:sizeof(int)   atIndex:10];
    [enc setBytes:&p_Height          length:sizeof(int)   atIndex:11];
    [enc setBytes:&p_ScreenColor     length:sizeof(int)   atIndex:12];
    [enc setBytes:&p_UseScreenInput  length:sizeof(int)   atIndex:13];
    [enc setBytes:&p_PickR           length:sizeof(float) atIndex:14];
    [enc setBytes:&p_PickG           length:sizeof(float) atIndex:15];
    [enc setBytes:&p_PickB           length:sizeof(float) atIndex:16];
    [enc setBytes:&p_Bias            length:sizeof(float) atIndex:17];
    [enc setBytes:&p_Limit           length:sizeof(float) atIndex:18];
    [enc setBytes:&p_RespillR        length:sizeof(float) atIndex:19];
    [enc setBytes:&p_RespillG        length:sizeof(float) atIndex:20];
    [enc setBytes:&p_RespillB        length:sizeof(float) atIndex:21];
    [enc setBytes:&p_NearGreyExtract length:sizeof(int)   atIndex:22];
    [enc setBytes:&p_NearGreyAmount  length:sizeof(float) atIndex:23];
    [enc setBytes:&p_NearGreySoftness length:sizeof(float) atIndex:28];
    [enc setBytes:&p_BlackClip       length:sizeof(float) atIndex:24];
    [enc setBytes:&p_WhiteClip       length:sizeof(float) atIndex:25];
    [enc setBytes:&p_EdgeProtect     length:sizeof(float) atIndex:26];
    [enc setBytes:&p_MatteGamma      length:sizeof(float) atIndex:27];
    dispatch2D(enc, pipes.coreKeyer, p_Width, p_Height);

    // After core: dstBuf=RGBA(despilled+rawAlpha), tempA=rawAlpha, tempB=guide

    if (doGF) {
        // ══════════════════════════════════════════════════════════════════
        //  GUIDED FILTER (iterative refinement)
        // ══════════════════════════════════════════════════════════════════
        int r = p_GuidedRadius;
        int numIter = std::max(1, std::min(p_RefineIterations, 5));

        // Save the original raw alpha to tempG for final mix
        [enc setComputePipelineState:pipes.copyBuffer];
        [enc setBuffer:tempA offset:0 atIndex:3];
        [enc setBuffer:tempG offset:0 atIndex:4];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        dispatch2D(enc, pipes.copyBuffer, p_Width, p_Height);

        for (int iter = 0; iter < numIter; iter++) {
            bool isLast = (iter == numIter - 1);

            if (iter > 0) {
                // Refine guide: use refined alpha + source RGB to make a better guide
                [enc setComputePipelineState:pipes.refineGuide];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:tempA  offset:0 atIndex:3];   // refined alpha
                [enc setBuffer:tempE  offset:0 atIndex:5];   // write: new guide
                [enc setBytes:&p_Width       length:sizeof(int)   atIndex:10];
                [enc setBytes:&p_Height      length:sizeof(int)   atIndex:11];
                [enc setBytes:&p_EdgeProtect length:sizeof(float) atIndex:12];
                dispatch2D(enc, pipes.refineGuide, p_Width, p_Height);
                // tempE = updated guide, tempA = p (refined alpha from prev iter)
            } else {
                // First iteration: copy initial guide from tempB → tempE
                [enc setComputePipelineState:pipes.copyBuffer];
                [enc setBuffer:tempB offset:0 atIndex:3];
                [enc setBuffer:tempE offset:0 atIndex:4];
                [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
                [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
                dispatch2D(enc, pipes.copyBuffer, p_Width, p_Height);
            }

            // Compute products: p(tempA), guide(tempE) → I(tempB), Ip(tempC), II(tempD)
            [enc setComputePipelineState:pipes.computeProducts];
            [enc setBuffer:tempA offset:0 atIndex:3];  // p
            [enc setBuffer:tempB offset:0 atIndex:4];  // output: copy of I
            [enc setBuffer:tempE offset:0 atIndex:5];  // guide (read-only)
            [enc setBuffer:tempC offset:0 atIndex:6];  // output: I*p
            [enc setBuffer:tempD offset:0 atIndex:7];  // output: I*I
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
            dispatch2D(enc, pipes.computeProducts, p_Width, p_Height);

            // Blur all four: true Gaussian, single H+V pass
            gaussianBlur(enc, pipes, tempA, tempF, weightBuf, p_Width, p_Height, r);
            gaussianBlur(enc, pipes, tempB, tempF, weightBuf, p_Width, p_Height, r);
            gaussianBlur(enc, pipes, tempC, tempF, weightBuf, p_Width, p_Height, r);
            gaussianBlur(enc, pipes, tempD, tempF, weightBuf, p_Width, p_Height, r);

            // Compute coefficients: a, b
            [enc setComputePipelineState:pipes.guidedCoeff];
            [enc setBuffer:tempB offset:0 atIndex:3];  // mean_I → a
            [enc setBuffer:tempA offset:0 atIndex:4];  // mean_p → b
            [enc setBuffer:tempC offset:0 atIndex:5];  // mean_Ip
            [enc setBuffer:tempD offset:0 atIndex:6];  // mean_II
            [enc setBytes:&p_Width          length:sizeof(int)   atIndex:10];
            [enc setBytes:&p_Height         length:sizeof(int)   atIndex:11];
            [enc setBytes:&p_GuidedEpsilon  length:sizeof(float) atIndex:12];
            dispatch2D(enc, pipes.guidedCoeff, p_Width, p_Height);

            // Blur a and b → mean_a, mean_b
            gaussianBlur(enc, pipes, tempB, tempF, weightBuf, p_Width, p_Height, r);
            gaussianBlur(enc, pipes, tempA, tempF, weightBuf, p_Width, p_Height, r);

            if (isLast) {
                // Final: apply with mix against saved raw alpha + premultiply
                [enc setComputePipelineState:pipes.guidedApply];
                [enc setBuffer:dstBuf offset:0 atIndex:2];
                [enc setBuffer:tempB  offset:0 atIndex:3];  // mean_a
                [enc setBuffer:tempA  offset:0 atIndex:4];  // mean_b
                [enc setBuffer:tempE  offset:0 atIndex:5];  // guide
                [enc setBuffer:tempG  offset:0 atIndex:6];  // saved raw alpha
                [enc setBytes:&p_Width       length:sizeof(int)   atIndex:10];
                [enc setBytes:&p_Height      length:sizeof(int)   atIndex:11];
                [enc setBytes:&p_Premultiply length:sizeof(int)   atIndex:12];
                [enc setBytes:&p_GuidedMix   length:sizeof(float) atIndex:13];
                dispatch2D(enc, pipes.guidedApply, p_Width, p_Height);
            } else {
                // Intermediate: evaluate refined alpha → tempC, then copy → tempA
                [enc setComputePipelineState:pipes.guidedEval];
                [enc setBuffer:tempB offset:0 atIndex:3];  // mean_a
                [enc setBuffer:tempA offset:0 atIndex:4];  // mean_b
                [enc setBuffer:tempE offset:0 atIndex:5];  // guide
                [enc setBuffer:tempC offset:0 atIndex:6];  // output: refined alpha
                [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
                [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
                dispatch2D(enc, pipes.guidedEval, p_Width, p_Height);

                // Copy tempC → tempA for next iteration's p
                [enc setComputePipelineState:pipes.copyBuffer];
                [enc setBuffer:tempC offset:0 atIndex:3];
                [enc setBuffer:tempA offset:0 atIndex:4];
                [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
                [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
                dispatch2D(enc, pipes.copyBuffer, p_Width, p_Height);
            }
        } // iteration loop

    } else if (p_Premultiply) {
        // No GF — just premultiply
        [enc setComputePipelineState:pipes.premultiply];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        dispatch2D(enc, pipes.premultiply, p_Width, p_Height);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PASS 2.5: Edge Color Correction
    //  Re-estimates FG color at semi-transparent edges using the matting
    //  equation: fg = (src - screen*(1-alpha)) / alpha
    // ══════════════════════════════════════════════════════════════════════
    if (p_EdgeColorCorrect > 0.0f) {
        [enc setComputePipelineState:pipes.edgeColorCorrect];
        [enc setBuffer:srcBuf offset:0 atIndex:0];
        [enc setBuffer:scrBuf offset:0 atIndex:1];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width            length:sizeof(int)   atIndex:10];
        [enc setBytes:&p_Height           length:sizeof(int)   atIndex:11];
        [enc setBytes:&p_UseScreenInput   length:sizeof(int)   atIndex:12];
        [enc setBytes:&p_PickR            length:sizeof(float) atIndex:13];
        [enc setBytes:&p_PickG            length:sizeof(float) atIndex:14];
        [enc setBytes:&p_PickB            length:sizeof(float) atIndex:15];
        [enc setBytes:&p_EdgeColorCorrect length:sizeof(float) atIndex:16];
        [enc setBytes:&p_Premultiply      length:sizeof(int)   atIndex:17];
        dispatch2D(enc, pipes.edgeColorCorrect, p_Width, p_Height);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PASS 3: Background Wrap
    //  Blurs the BG and bleeds it into FG edges weighted by (1-alpha)
    // ══════════════════════════════════════════════════════════════════════
    if (doBgWrap) {
        id<MTLBuffer> bgBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Background));

        // Pre-compute Gaussian weights for BG blur
        int bwR = std::max(1, p_BgWrapBlur);
        int bwKernelSize = 2 * bwR + 1;
        float bwSigma = fmaxf(bwR / 3.0f, 0.5f);
        float bwInv2s2 = 1.0f / (2.0f * bwSigma * bwSigma);
        float* bwW = (float*)alloca(bwKernelSize * sizeof(float));
        float bwSum = 0.0f;
        for (int i = -bwR; i <= bwR; i++) {
            float wt = expf(-(float)(i * i) * bwInv2s2);
            bwW[i + bwR] = wt;
            bwSum += wt;
        }
        for (int i = 0; i < bwKernelSize; i++) bwW[i] /= bwSum;

        id<MTLBuffer> bwWeightBuf = [device newBufferWithBytes:bwW
                                                        length:bwKernelSize * sizeof(float)
                                                       options:MTLResourceStorageModeShared];

        // Extract R, G, B from BG into tempA, tempB, tempC
        for (int ch = 0; ch < 3; ch++) {
            id<MTLBuffer> dst = (ch == 0) ? tempA : (ch == 1) ? tempB : tempC;
            [enc setComputePipelineState:pipes.extractChannel];
            [enc setBuffer:bgBuf offset:0 atIndex:0];
            [enc setBuffer:dst   offset:0 atIndex:3];
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
            [enc setBytes:&ch       length:sizeof(int) atIndex:12];
            dispatch2D(enc, pipes.extractChannel, p_Width, p_Height);
        }

        // Gaussian blur each channel (reuse tempF as scratch)
        gaussianBlur(enc, pipes, tempA, tempF, bwWeightBuf, p_Width, p_Height, bwR);
        gaussianBlur(enc, pipes, tempB, tempF, bwWeightBuf, p_Width, p_Height, bwR);
        gaussianBlur(enc, pipes, tempC, tempF, bwWeightBuf, p_Width, p_Height, bwR);

        // Apply wrap
        [enc setComputePipelineState:pipes.bgWrap];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBuffer:tempA  offset:0 atIndex:3];
        [enc setBuffer:tempB  offset:0 atIndex:4];
        [enc setBuffer:tempC  offset:0 atIndex:5];
        [enc setBytes:&p_Width       length:sizeof(int)   atIndex:10];
        [enc setBytes:&p_Height      length:sizeof(int)   atIndex:11];
        [enc setBytes:&p_BgWrapAmount length:sizeof(float) atIndex:12];
        dispatch2D(enc, pipes.bgWrap, p_Width, p_Height);

        [bwWeightBuf release];
    }

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // ── Cleanup (temp buffers are cached — only release per-frame objects) ──
    if (createdDummy) [scrBuf release];
    if (weightBuf) [weightBuf release];
  } // @autoreleasepool
}
