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
//  Gaussian Blur — Horizontal TILED (threadgroup shared memory)
//  Each row of the threadgroup cooperatively loads a strip into fast
//  shared memory, then each thread sums from the tile.  Dramatically
//  reduces global-memory bandwidth for large radii.
//  Static 4096-float tile supports radius up to ~240 with (32,8) tg.
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurH_Tiled(
    const device float* p_Src     [[buffer(3)]],
    device float*       p_Dst     [[buffer(4)]],
    const device float* p_Weights [[buffer(5)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Radius  [[buffer(12)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgs  [[threads_per_threadgroup]])
{
    threadgroup float tile[4096];

    int x = (int)gid.x, y = (int)gid.y;
    if (y >= p_Height) return;

    int r = p_Radius;
    int tileW = (int)tgs.x + 2 * r;
    int rowOff = (int)tid.y * tileW;
    int baseX = x - (int)tid.x - r;

    for (int i = (int)tid.x; i < tileW; i += (int)tgs.x) {
        int sx = clamp(baseX + i, 0, p_Width - 1);
        tile[rowOff + i] = p_Src[y * p_Width + sx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (x >= p_Width) return;

    float sum = 0.0f;
    int center = rowOff + (int)tid.x + r;
    for (int dx = -r; dx <= r; dx++) {
        sum += tile[center + dx] * p_Weights[dx + r];
    }
    p_Dst[y * p_Width + x] = sum;
}

// ════════════════════════════════════════════════════════════════════════
//  Gaussian Blur — Vertical TILED (threadgroup shared memory)
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurV_Tiled(
    const device float* p_Src     [[buffer(3)]],
    device float*       p_Dst     [[buffer(4)]],
    const device float* p_Weights [[buffer(5)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Radius  [[buffer(12)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgs  [[threads_per_threadgroup]])
{
    threadgroup float tile[4096];

    int x = (int)gid.x, y = (int)gid.y;
    if (x >= p_Width) return;

    int r = p_Radius;
    int tileH = (int)tgs.y + 2 * r;
    int colOff = (int)tid.x * tileH;
    int baseY = y - (int)tid.y - r;

    for (int i = (int)tid.y; i < tileH; i += (int)tgs.y) {
        int sy = clamp(baseY + i, 0, p_Height - 1);
        tile[colOff + i] = p_Src[sy * p_Width + x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (y >= p_Height) return;

    float sum = 0.0f;
    int center = colOff + (int)tid.y + r;
    for (int dy = -r; dy <= r; dy++) {
        sum += tile[center + dy] * p_Weights[dy + r];
    }
    p_Dst[y * p_Width + x] = sum;
}

// ════════════════════════════════════════════════════════════════════════
//  Gaussian Blur — Horizontal, 4-channel (processes 4 separate buffers)
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurH4(
    const device float* p_A [[buffer(0)]],
    const device float* p_B [[buffer(1)]],
    const device float* p_C [[buffer(2)]],
    const device float* p_D [[buffer(3)]],
    device float* p_OA [[buffer(4)]],
    device float* p_OB [[buffer(5)]],
    device float* p_OC [[buffer(6)]],
    device float* p_OD [[buffer(7)]],
    const device float* p_Weights [[buffer(8)]],
    constant int& p_Width  [[buffer(10)]],
    constant int& p_Height [[buffer(11)]],
    constant int& p_Radius [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    int x = (int)id.x, y = (int)id.y, r = p_Radius, w = p_Width;
    float sA = 0.0f, sB = 0.0f, sC = 0.0f, sD = 0.0f;
    for (int dx = -r; dx <= r; dx++) {
        int sx = clamp(x + dx, 0, w - 1);
        int si = y * w + sx;
        float wt = p_Weights[dx + r];
        sA += p_A[si] * wt;
        sB += p_B[si] * wt;
        sC += p_C[si] * wt;
        sD += p_D[si] * wt;
    }
    int oi = y * w + x;
    p_OA[oi] = sA; p_OB[oi] = sB; p_OC[oi] = sC; p_OD[oi] = sD;
}

// ════════════════════════════════════════════════════════════════════════
//  Gaussian Blur — Vertical, 4-channel (processes 4 separate buffers)
// ════════════════════════════════════════════════════════════════════════

kernel void GaussianBlurV4(
    const device float* p_A [[buffer(0)]],
    const device float* p_B [[buffer(1)]],
    const device float* p_C [[buffer(2)]],
    const device float* p_D [[buffer(3)]],
    device float* p_OA [[buffer(4)]],
    device float* p_OB [[buffer(5)]],
    device float* p_OC [[buffer(6)]],
    device float* p_OD [[buffer(7)]],
    const device float* p_Weights [[buffer(8)]],
    constant int& p_Width  [[buffer(10)]],
    constant int& p_Height [[buffer(11)]],
    constant int& p_Radius [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    int x = (int)id.x, y = (int)id.y, r = p_Radius, w = p_Width, h = p_Height;
    float sA = 0.0f, sB = 0.0f, sC = 0.0f, sD = 0.0f;
    for (int dy = -r; dy <= r; dy++) {
        int sy = clamp(y + dy, 0, h - 1);
        int si = sy * w + x;
        float wt = p_Weights[dy + r];
        sA += p_A[si] * wt;
        sB += p_B[si] * wt;
        sC += p_C[si] * wt;
        sD += p_D[si] * wt;
    }
    int oi = y * w + x;
    p_OA[oi] = sA; p_OB[oi] = sB; p_OC[oi] = sC; p_OD[oi] = sD;
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

    // Adaptive epsilon: scales down at edges (high variance) for better
    // edge preservation, stays full-strength in flat regions to suppress noise.
    // adaptEps = eps^2 / (varI + eps), so:
    //   flat (varI≈0): adaptEps ≈ eps  → smooths normally
    //   edge (varI>>eps): adaptEps ≈ eps^2/varI → tiny → preserves edges
    float adaptEps = p_Epsilon * p_Epsilon / (varI + p_Epsilon + 1e-10f);
    float a = covIp / (varI + adaptEps);
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
//  Additive Key — recovers fine detail (hair, motion blur, transparency)
//  the alpha-based key lost, by superimposing source-minus-screen onto
//  the composite.  Two paths:
//    Addition:       residual = src - screen  →  desaturate  →  add
//    Multiplication: factor = src/screen - 1  →  desaturate  →  × BG
//  Weighted by (1-alpha) so only transparent areas are affected.
// ════════════════════════════════════════════════════════════════════════

kernel void AdditiveKeyKernel(
    const device float* p_Source   [[buffer(0)]],
    const device float* p_Screen   [[buffer(1)]],
    device float*       p_Output   [[buffer(2)]],
    const device float* p_BlurBgR  [[buffer(3)]],
    const device float* p_BlurBgG  [[buffer(4)]],
    const device float* p_BlurBgB  [[buffer(5)]],
    constant int&       p_Width    [[buffer(10)]],
    constant int&       p_Height   [[buffer(11)]],
    constant int&       p_Mode     [[buffer(12)]],
    constant int&       p_UseScr   [[buffer(13)]],
    constant float&     p_ScrR     [[buffer(14)]],
    constant float&     p_ScrG     [[buffer(15)]],
    constant float&     p_ScrB     [[buffer(16)]],
    constant float&     p_Sat      [[buffer(17)]],
    constant float&     p_Amount   [[buffer(18)]],
    constant int&       p_ClampBlk [[buffer(19)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;

    float alpha = p_Output[idx4 + 3];
    if (alpha > 0.999f) return;  // solid FG — standard key handles it

    float srcR = p_Source[idx4 + 0];
    float srcG = p_Source[idx4 + 1];
    float srcB = p_Source[idx4 + 2];

    float scrR, scrG, scrB;
    if (p_UseScr != 0) {
        scrR = p_Screen[idx4 + 0];
        scrG = p_Screen[idx4 + 1];
        scrB = p_Screen[idx4 + 2];
    } else {
        scrR = p_ScrR; scrG = p_ScrG; scrB = p_ScrB;
    }

    float resR, resG, resB;

    if (p_Mode == 0) {
        // ── Addition path: source - screen ──
        resR = srcR - scrR;
        resG = srcG - scrG;
        resB = srcB - scrB;
        // Desaturate to remove color cast
        float lum = 0.2126f * resR + 0.7152f * resG + 0.0722f * resB;
        resR = mix(lum, resR, p_Sat);
        resG = mix(lum, resG, p_Sat);
        resB = mix(lum, resB, p_Sat);
    } else {
        // ── Multiplication path: (source/screen) × BG ──
        float fR = (scrR > 1e-6f) ? srcR / scrR : 1.0f;
        float fG = (scrG > 1e-6f) ? srcG / scrG : 1.0f;
        float fB = (scrB > 1e-6f) ? srcB / scrB : 1.0f;
        // Desaturate factor (neutral = 1.0)
        float fLum = 0.2126f * fR + 0.7152f * fG + 0.0722f * fB;
        fR = mix(fLum, fR, p_Sat);
        fG = mix(fLum, fG, p_Sat);
        fB = mix(fLum, fB, p_Sat);
        // Delta from original BG: BG × (factor-1)
        resR = p_BlurBgR[idx1] * (fR - 1.0f);
        resG = p_BlurBgG[idx1] * (fG - 1.0f);
        resB = p_BlurBgB[idx1] * (fB - 1.0f);
    }

    // Black clamp (optional — keeps only brighter-than-screen detail)
    if (p_ClampBlk != 0) {
        resR = max(resR, 0.0f);
        resG = max(resG, 0.0f);
        resB = max(resB, 0.0f);
    }

    // Weight by (1-alpha) · amount and add to output
    float w = (1.0f - alpha) * p_Amount;
    p_Output[idx4 + 0] += resR * w;
    p_Output[idx4 + 1] += resG * w;
    p_Output[idx4 + 2] += resB * w;
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

// ════════════════════════════════════════════════════════════════════════
//  RGB Guided Filter — Compute Products
//  Extracts RGB guide channels from source, copies alpha, and computes
//  all 13 statistical channels needed for the 3-channel guided filter:
//    4 means (Ir, Ig, Ib, p)  +  6 auto-covariance (IrIr..IbIb)
//    + 3 cross-covariance (Irp, Igp, Ibp)
// ════════════════════════════════════════════════════════════════════════

kernel void RGBComputeProducts(
    const device float* p_Input   [[buffer(0)]],   // source RGBA
    const device float* p_Output  [[buffer(1)]],   // output from core keyer (alpha in .w)
    device float* p_MeanIr   [[buffer(2)]],
    device float* p_MeanIg   [[buffer(3)]],
    device float* p_MeanIb   [[buffer(4)]],
    device float* p_MeanP    [[buffer(5)]],
    device float* p_IrIr     [[buffer(6)]],
    device float* p_IrIg     [[buffer(7)]],
    device float* p_IrIb     [[buffer(8)]],
    device float* p_IgIg     [[buffer(9)]],
    device float* p_IgIb     [[buffer(10)]],
    device float* p_IbIb     [[buffer(11)]],
    device float* p_IrP      [[buffer(12)]],
    device float* p_IgP      [[buffer(13)]],
    device float* p_IbP      [[buffer(14)]],
    constant int& p_Width    [[buffer(20)]],
    constant int& p_Height   [[buffer(21)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float Ir = p_Input[idx4 + 0];
    float Ig = p_Input[idx4 + 1];
    float Ib = p_Input[idx4 + 2];
    float p  = p_Output[idx4 + 3];  // alpha from core keyer

    // Copies for blurring (will become means)
    p_MeanIr[idx1] = Ir;
    p_MeanIg[idx1] = Ig;
    p_MeanIb[idx1] = Ib;
    p_MeanP[idx1]  = p;

    // Auto-covariance products
    p_IrIr[idx1] = Ir * Ir;
    p_IrIg[idx1] = Ir * Ig;
    p_IrIb[idx1] = Ir * Ib;
    p_IgIg[idx1] = Ig * Ig;
    p_IgIb[idx1] = Ig * Ib;
    p_IbIb[idx1] = Ib * Ib;

    // Cross-covariance products
    p_IrP[idx1] = Ir * p;
    p_IgP[idx1] = Ig * p;
    p_IbP[idx1] = Ib * p;
}

// ════════════════════════════════════════════════════════════════════════
//  RGB Guided Filter — Coefficient Computation
//  Reads 13 blurred channels, solves the 3×3 system:
//    (Σ + εI) · a = cov(I, p)
//    b = mean_p − aᵀ · mean_I
//  Writes a_r, a_g, a_b, b into 4 output buffers.
// ════════════════════════════════════════════════════════════════════════

kernel void RGBGuidedCoeff(
    const device float* p_MeanIr  [[buffer(0)]],
    const device float* p_MeanIg  [[buffer(1)]],
    const device float* p_MeanIb  [[buffer(2)]],
    const device float* p_MeanP   [[buffer(3)]],
    device float*       p_IrIr    [[buffer(4)]],   // overwritten with a_r
    const device float* p_IrIg    [[buffer(5)]],
    const device float* p_IrIb    [[buffer(6)]],
    device float*       p_IgIg    [[buffer(7)]],   // overwritten with a_g
    const device float* p_IgIb    [[buffer(8)]],
    device float*       p_IbIb    [[buffer(9)]],   // overwritten with a_b
    const device float* p_IrP     [[buffer(10)]],
    const device float* p_IgP     [[buffer(11)]],
    const device float* p_IbP     [[buffer(12)]],
    device float*       p_OutB    [[buffer(13)]],   // output b coefficient
    constant int&       p_Width   [[buffer(20)]],
    constant int&       p_Height  [[buffer(21)]],
    constant float&     p_Epsilon [[buffer(22)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int i = (int)(id.y * (uint)p_Width) + (int)id.x;

    float mIr = p_MeanIr[i], mIg = p_MeanIg[i], mIb = p_MeanIb[i], mP = p_MeanP[i];

    // Covariance matrix Σ (symmetric 3×3)
    float s_rr = p_IrIr[i] - mIr * mIr;
    float s_rg = p_IrIg[i] - mIr * mIg;
    float s_rb = p_IrIb[i] - mIr * mIb;
    float s_gg = p_IgIg[i] - mIg * mIg;
    float s_gb = p_IgIb[i] - mIg * mIb;
    float s_bb = p_IbIb[i] - mIb * mIb;

    // Cross-covariance cov(I, p)
    float c_rp = p_IrP[i] - mIr * mP;
    float c_gp = p_IgP[i] - mIg * mP;
    float c_bp = p_IbP[i] - mIb * mP;

    // Adaptive epsilon: same approach as scalar but applied to trace
    float trace = s_rr + s_gg + s_bb;
    float adaptEps = p_Epsilon * p_Epsilon / (trace / 3.0f + p_Epsilon + 1e-10f);

    // Add ε·I to diagonal
    s_rr += adaptEps;
    s_gg += adaptEps;
    s_bb += adaptEps;

    // Solve 3×3 symmetric system via Cramer's rule
    // det(M)
    float det = s_rr * (s_gg * s_bb - s_gb * s_gb)
              - s_rg * (s_rg * s_bb - s_gb * s_rb)
              + s_rb * (s_rg * s_gb - s_gg * s_rb);

    float invDet = (abs(det) > 1e-12f) ? (1.0f / det) : 0.0f;

    // Cofactor matrix (symmetric) for inverse
    float inv_rr = (s_gg * s_bb - s_gb * s_gb) * invDet;
    float inv_rg = (s_rb * s_gb - s_rg * s_bb) * invDet;
    float inv_rb = (s_rg * s_gb - s_rb * s_gg) * invDet;
    float inv_gg = (s_rr * s_bb - s_rb * s_rb) * invDet;
    float inv_gb = (s_rb * s_rg - s_rr * s_gb) * invDet;
    float inv_bb = (s_rr * s_gg - s_rg * s_rg) * invDet;

    // a = inv(Σ+εI) · cov(I,p)
    float ar = inv_rr * c_rp + inv_rg * c_gp + inv_rb * c_bp;
    float ag = inv_rg * c_rp + inv_gg * c_gp + inv_gb * c_bp;
    float ab = inv_rb * c_rp + inv_gb * c_gp + inv_bb * c_bp;
    float b  = mP - ar * mIr - ag * mIg - ab * mIb;

    // Write coefficients (reuse 4 buffers)
    p_IrIr[i] = ar;     // buf[4] = a_r
    p_IgIg[i] = ag;     // buf[7] = a_g
    p_IbIb[i] = ab;     // buf[9] = a_b
    p_OutB[i] = b;      // buf[13] = b
}

// ════════════════════════════════════════════════════════════════════════
//  RGB Guided Filter — Intermediate Evaluation
//  For iterative refinement: compute refined alpha without premultiply.
//  q = mean_ar * Ir + mean_ag * Ig + mean_ab * Ib + mean_b
// ════════════════════════════════════════════════════════════════════════

kernel void RGBGuidedEval(
    const device float* p_Input   [[buffer(0)]],   // source RGBA (for RGB guide)
    const device float* p_MeanAr  [[buffer(2)]],
    const device float* p_MeanAg  [[buffer(3)]],
    const device float* p_MeanAb  [[buffer(4)]],
    const device float* p_MeanB   [[buffer(5)]],
    device float*       p_OutAlpha [[buffer(6)]],
    constant int&       p_Width   [[buffer(20)]],
    constant int&       p_Height  [[buffer(21)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float Ir = p_Input[idx4 + 0];
    float Ig = p_Input[idx4 + 1];
    float Ib = p_Input[idx4 + 2];

    float q = p_MeanAr[idx1] * Ir + p_MeanAg[idx1] * Ig
            + p_MeanAb[idx1] * Ib + p_MeanB[idx1];
    p_OutAlpha[idx1] = clamp(q, 0.0f, 1.0f);
}

// ════════════════════════════════════════════════════════════════════════
//  RGB Guided Filter — Final Apply
//  Evaluates q, mixes with raw alpha, writes RGBA output + premultiply.
// ════════════════════════════════════════════════════════════════════════

kernel void RGBGuidedApply(
    const device float* p_Input    [[buffer(0)]],   // source RGBA (for RGB guide)
    device float*       p_Output   [[buffer(1)]],   // output RGBA (read+write)
    const device float* p_MeanAr   [[buffer(2)]],
    const device float* p_MeanAg   [[buffer(3)]],
    const device float* p_MeanAb   [[buffer(4)]],
    const device float* p_MeanB    [[buffer(5)]],
    const device float* p_RawAlpha [[buffer(6)]],
    constant int&       p_Width    [[buffer(20)]],
    constant int&       p_Height   [[buffer(21)]],
    constant int&       p_Premultiply [[buffer(22)]],
    constant float&     p_GFMix    [[buffer(23)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float Ir = p_Input[idx4 + 0];
    float Ig = p_Input[idx4 + 1];
    float Ib = p_Input[idx4 + 2];

    float rawAlpha = p_RawAlpha[idx1];
    float guidedAlpha = clamp(
        p_MeanAr[idx1] * Ir + p_MeanAg[idx1] * Ig
        + p_MeanAb[idx1] * Ib + p_MeanB[idx1], 0.0f, 1.0f);
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
//  Write Alpha — copies 1ch alpha buffer to RGBA alpha channel
//  Used during iterative RGB guided filter refinement.
// ════════════════════════════════════════════════════════════════════════

kernel void WriteAlphaKernel(
    const device float* p_Alpha  [[buffer(0)]],
    device float*       p_RGBA   [[buffer(1)]],
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    p_RGBA[idx1 * 4 + 3] = p_Alpha[idx1];
}

// ════════════════════════════════════════════════════════════════════════
//  Erode Alpha — morphological minimum filter on 1ch alpha
//  Shrinks the initial matte before clean plate estimation to prevent
//  foreground contamination in the synthetic screen.
// ════════════════════════════════════════════════════════════════════════

kernel void ErodeAlphaKernel(
    const device float* p_Src    [[buffer(0)]],
    device float*       p_Dst    [[buffer(1)]],
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    constant int&       p_Radius [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    int x = (int)id.x, y = (int)id.y, r = p_Radius;
    int w = p_Width, h = p_Height;
    float minVal = 1.0f;
    for (int dy = -r; dy <= r; dy++) {
        int sy = clamp(y + dy, 0, h - 1);
        for (int dx = -r; dx <= r; dx++) {
            int sx = clamp(x + dx, 0, w - 1);
            minVal = min(minVal, p_Src[sy * w + sx]);
        }
    }
    p_Dst[y * w + x] = minVal;
}

// ════════════════════════════════════════════════════════════════════════
//  Clean Plate Estimate — IBKColour-style synthetic screen generation
//  Where alpha ≈ 0 (pure screen): keeps source pixel (preserves screen
//  variation — light falloff, wrinkles, color gradients)
//  Where alpha ≈ 1 (pure FG): replaces with picked screen color
//  Smoothstep blending avoids hard transition artifacts.
//  The subsequent blur fills FG holes from surrounding BG information.
// ════════════════════════════════════════════════════════════════════════

kernel void CleanPlateEstimateKernel(
    const device float* p_Input  [[buffer(0)]],   // source RGBA
    const device float* p_Alpha  [[buffer(1)]],   // 1ch initial alpha (eroded)
    device float*       p_OutR   [[buffer(2)]],   // clean plate R channel
    device float*       p_OutG   [[buffer(3)]],   // clean plate G channel
    device float*       p_OutB   [[buffer(4)]],   // clean plate B channel
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    constant float&     p_PickR  [[buffer(12)]],
    constant float&     p_PickG  [[buffer(13)]],
    constant float&     p_PickB  [[buffer(14)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx4 = ((int)(id.y * (uint)p_Width) + (int)id.x) * 4;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;

    float srcR = p_Input[idx4 + 0];
    float srcG = p_Input[idx4 + 1];
    float srcB = p_Input[idx4 + 2];
    float alpha = clamp(p_Alpha[idx1], 0.0f, 1.0f);

    // Smoothstep for softer transition
    float t = alpha * alpha * (3.0f - 2.0f * alpha);

    // Blend: keep source at alpha=0 (screen), replace with picked color at alpha=1 (FG)
    p_OutR[idx1] = mix(srcR, p_PickR, t);
    p_OutG[idx1] = mix(srcG, p_PickG, t);
    p_OutB[idx1] = mix(srcB, p_PickB, t);
}

// ════════════════════════════════════════════════════════════════════════
//  Pack RGBA — assembles 3 float channels into an interleaved RGBA buffer
//  Builds the clean plate RGBA for the second core key pass.
// ════════════════════════════════════════════════════════════════════════

kernel void PackRGBAKernel(
    const device float* p_R      [[buffer(0)]],
    const device float* p_G      [[buffer(1)]],
    const device float* p_B      [[buffer(2)]],
    device float*       p_RGBA   [[buffer(3)]],
    constant int&       p_Width  [[buffer(10)]],
    constant int&       p_Height [[buffer(11)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;
    p_RGBA[idx4 + 0] = p_R[idx1];
    p_RGBA[idx4 + 1] = p_G[idx1];
    p_RGBA[idx4 + 2] = p_B[idx1];
    p_RGBA[idx4 + 3] = 1.0f;
}

// ════════════════════════════════════════════════════════════════════════
//  Apply External Matte — garbage / occlusion matte support
//  Reads alpha from an RGBA matte buffer and modifies both the 1ch alpha
//  and the output RGBA buffer's alpha channel.
//  Mode 0 = garbage (white = remove: alpha *= 1 - matte)
//  Mode 1 = occlusion (white = keep:  alpha = max(alpha, matte))
// ════════════════════════════════════════════════════════════════════════

kernel void ApplyMatteKernel(
    device float*       p_Alpha   [[buffer(0)]],
    device float*       p_Output  [[buffer(1)]],
    const device float* p_Matte   [[buffer(2)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Mode    [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;

    float matteVal = p_Matte[idx4 + 3];
    float alpha = p_Alpha[idx1];

    if (p_Mode == 0) {
        // Garbage: white areas = remove from key
        alpha *= (1.0f - matteVal);
    } else {
        // Occlusion: white areas = force opaque
        alpha = max(alpha, matteVal);
    }

    p_Alpha[idx1] = alpha;
    p_Output[idx4 + 3] = alpha;
}

// ════════════════════════════════════════════════════════════════════════
//  Diagnostic Output — writes intermediate pipeline data to output
//  Mode 0: 1-channel alpha buffer → greyscale RGBA (R=G=B=alpha, A=1)
//  Mode 1: copy RGBA buffer to output
//  Mode 2: extract alpha from RGBA → greyscale (R=G=B=src.a, A=1)
// ════════════════════════════════════════════════════════════════════════

kernel void DiagnosticOutputKernel(
    const device float* p_SrcA    [[buffer(0)]],
    const device float* p_SrcRGBA [[buffer(1)]],
    device float*       p_Output  [[buffer(2)]],
    constant int&       p_Width   [[buffer(10)]],
    constant int&       p_Height  [[buffer(11)]],
    constant int&       p_Mode    [[buffer(12)]],
    uint2 id [[thread_position_in_grid]])
{
    if ((int)id.x >= p_Width || (int)id.y >= p_Height) return;
    const int idx1 = (int)(id.y * (uint)p_Width) + (int)id.x;
    const int idx4 = idx1 * 4;

    if (p_Mode == 0) {
        // 1-channel alpha → greyscale RGBA
        float a = p_SrcA[idx1];
        p_Output[idx4 + 0] = a;
        p_Output[idx4 + 1] = a;
        p_Output[idx4 + 2] = a;
        p_Output[idx4 + 3] = 1.0f;
    } else if (p_Mode == 1) {
        // Copy RGBA
        p_Output[idx4 + 0] = p_SrcRGBA[idx4 + 0];
        p_Output[idx4 + 1] = p_SrcRGBA[idx4 + 1];
        p_Output[idx4 + 2] = p_SrcRGBA[idx4 + 2];
        p_Output[idx4 + 3] = p_SrcRGBA[idx4 + 3];
    } else {
        // Mode 2: extract alpha from RGBA → greyscale
        float a = p_SrcRGBA[idx4 + 3];
        p_Output[idx4 + 0] = a;
        p_Output[idx4 + 1] = a;
        p_Output[idx4 + 2] = a;
        p_Output[idx4 + 3] = 1.0f;
    }
}
)";  // end of kernelSource

struct PipelineSet {
    id<MTLComputePipelineState> coreKeyer;
    id<MTLComputePipelineState> gaussianBlurH;
    id<MTLComputePipelineState> gaussianBlurV;
    id<MTLComputePipelineState> gaussianBlurH4;
    id<MTLComputePipelineState> gaussianBlurV4;
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
    // RGB guided filter
    id<MTLComputePipelineState> rgbComputeProducts;
    id<MTLComputePipelineState> rgbGuidedCoeff;
    id<MTLComputePipelineState> rgbGuidedEval;
    id<MTLComputePipelineState> rgbGuidedApply;
    id<MTLComputePipelineState> writeAlpha;
    // Prematte (clean plate generation)
    id<MTLComputePipelineState> erodeAlpha;
    id<MTLComputePipelineState> cleanPlateEstimate;
    id<MTLComputePipelineState> packRGBA;
    // External mattes
    id<MTLComputePipelineState> applyMatte;
    // Diagnostic output
    id<MTLComputePipelineState> diagnosticOutput;
    // Tiled blur (threadgroup shared memory)
    id<MTLComputePipelineState> gaussianBlurHTiled;
    id<MTLComputePipelineState> gaussianBlurVTiled;
    // Additive key
    id<MTLComputePipelineState> additiveKey;
};

struct QueueState {
    PipelineSet pipes;
    // Temp buffer pool:
    //   Scalar GF: uses temp[0..6]
    //     [0]=rawAlpha/p, [1]=guide/I, [2]=Ip, [3]=II,
    //     [4]=guideCopy, [5]=blurScratch, [6]=savedRawAlpha
    //   RGB GF: uses temp[0..17]
    //     [0..3]=meanIr,Ig,Ib,P  [4..9]=IrIr,IrIg,IrIb,IgIg,IgIb,IbIb
    //     [10..12]=Irp,Igp,Ibp  [13..16]=blur4 scratch  [17]=savedRawAlpha
    static const int MAX_TEMP = 18;
    id<MTLBuffer> temp[MAX_TEMP] = {};
    size_t cachedChanBytes = 0;
    int cachedBufCount = 0;   // how many temps are currently allocated
    // Prematte clean plate buffer (RGBA-sized)
    id<MTLBuffer> cleanPlateBuf = nil;
    size_t cachedCleanPlateBytes = 0;
    // Cached Gaussian weight buffers (avoid per-frame alloc/free)
    id<MTLBuffer> gfWeightBuf = nil;   int gfWeightRadius = -1;
    id<MTLBuffer> pmWeightBuf = nil;   int pmWeightRadius = -1;
    id<MTLBuffer> bwWeightBuf = nil;   int bwWeightRadius = -1;
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
//  Uses threadgroup-tiled kernels when the tile fits in 4096-float shared
//  memory (radius ≤ 240), falls back to the simple global-read kernels
//  for very large radii.
// ═══════════════════════════════════════════════════════════════════════════════

static void gaussianBlur(id<MTLComputeCommandEncoder> enc,
                         const PipelineSet& ps,
                         id<MTLBuffer> bufA, id<MTLBuffer> bufScratch,
                         id<MTLBuffer> weightBuf,
                         int w, int h, int radius)
{
    // Threadgroup sizes: (32,8)=256 for H, (8,32)=256 for V
    const int tgW_H = 32, tgH_H = 8;
    const int tgW_V = 8,  tgH_V = 32;
    bool canTileH = (tgH_H * (tgW_H + 2 * radius) <= 4096);
    bool canTileV = (tgW_V * (tgH_V + 2 * radius) <= 4096);

    if (canTileH && canTileV && ps.gaussianBlurHTiled && ps.gaussianBlurVTiled) {
        // ── Tiled H: A → scratch ──
        [enc setComputePipelineState:ps.gaussianBlurHTiled];
        [enc setBuffer:bufA       offset:0 atIndex:3];
        [enc setBuffer:bufScratch offset:0 atIndex:4];
        [enc setBuffer:weightBuf  offset:0 atIndex:5];
        [enc setBytes:&w      length:sizeof(int) atIndex:10];
        [enc setBytes:&h      length:sizeof(int) atIndex:11];
        [enc setBytes:&radius length:sizeof(int) atIndex:12];
        MTLSize tgSizeH = MTLSizeMake(tgW_H, tgH_H, 1);
        MTLSize gridH   = MTLSizeMake((w + tgW_H - 1) / tgW_H, (h + tgH_H - 1) / tgH_H, 1);
        [enc dispatchThreadgroups:gridH threadsPerThreadgroup:tgSizeH];

        // ── Tiled V: scratch → A ──
        [enc setComputePipelineState:ps.gaussianBlurVTiled];
        [enc setBuffer:bufScratch offset:0 atIndex:3];
        [enc setBuffer:bufA       offset:0 atIndex:4];
        [enc setBuffer:weightBuf  offset:0 atIndex:5];
        [enc setBytes:&w      length:sizeof(int) atIndex:10];
        [enc setBytes:&h      length:sizeof(int) atIndex:11];
        [enc setBytes:&radius length:sizeof(int) atIndex:12];
        MTLSize tgSizeV = MTLSizeMake(tgW_V, tgH_V, 1);
        MTLSize gridV   = MTLSizeMake((w + tgW_V - 1) / tgW_V, (h + tgH_V - 1) / tgH_V, 1);
        [enc dispatchThreadgroups:gridV threadsPerThreadgroup:tgSizeV];
    } else {
        // ── Fallback: global-read kernels ──
        [enc setComputePipelineState:ps.gaussianBlurH];
        [enc setBuffer:bufA       offset:0 atIndex:3];
        [enc setBuffer:bufScratch offset:0 atIndex:4];
        [enc setBuffer:weightBuf  offset:0 atIndex:5];
        [enc setBytes:&w      length:sizeof(int) atIndex:10];
        [enc setBytes:&h      length:sizeof(int) atIndex:11];
        [enc setBytes:&radius length:sizeof(int) atIndex:12];
        dispatch2D(enc, ps.gaussianBlurH, w, h);

        [enc setComputePipelineState:ps.gaussianBlurV];
        [enc setBuffer:bufScratch offset:0 atIndex:3];
        [enc setBuffer:bufA       offset:0 atIndex:4];
        [enc setBuffer:weightBuf  offset:0 atIndex:5];
        [enc setBytes:&w      length:sizeof(int) atIndex:10];
        [enc setBytes:&h      length:sizeof(int) atIndex:11];
        [enc setBytes:&radius length:sizeof(int) atIndex:12];
        dispatch2D(enc, ps.gaussianBlurV, w, h);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Gaussian blur helper: 4-channel separable H+V in 2 dispatches.
//  Blurs a,b,c,d in-place using sa,sb,sc,sd as scratch.
// ═══════════════════════════════════════════════════════════════════════════════

static void gaussianBlur4(id<MTLComputeCommandEncoder> enc,
                          const PipelineSet& ps,
                          id<MTLBuffer> a, id<MTLBuffer> b,
                          id<MTLBuffer> c, id<MTLBuffer> d,
                          id<MTLBuffer> sa, id<MTLBuffer> sb,
                          id<MTLBuffer> sc, id<MTLBuffer> sd,
                          id<MTLBuffer> weightBuf,
                          int w, int h, int radius)
{
    // Horizontal: a,b,c,d → sa,sb,sc,sd
    [enc setComputePipelineState:ps.gaussianBlurH4];
    [enc setBuffer:a  offset:0 atIndex:0];
    [enc setBuffer:b  offset:0 atIndex:1];
    [enc setBuffer:c  offset:0 atIndex:2];
    [enc setBuffer:d  offset:0 atIndex:3];
    [enc setBuffer:sa offset:0 atIndex:4];
    [enc setBuffer:sb offset:0 atIndex:5];
    [enc setBuffer:sc offset:0 atIndex:6];
    [enc setBuffer:sd offset:0 atIndex:7];
    [enc setBuffer:weightBuf offset:0 atIndex:8];
    [enc setBytes:&w      length:sizeof(int) atIndex:10];
    [enc setBytes:&h      length:sizeof(int) atIndex:11];
    [enc setBytes:&radius length:sizeof(int) atIndex:12];
    dispatch2D(enc, ps.gaussianBlurH4, w, h);

    // Vertical: sa,sb,sc,sd → a,b,c,d
    [enc setComputePipelineState:ps.gaussianBlurV4];
    [enc setBuffer:sa offset:0 atIndex:0];
    [enc setBuffer:sb offset:0 atIndex:1];
    [enc setBuffer:sc offset:0 atIndex:2];
    [enc setBuffer:sd offset:0 atIndex:3];
    [enc setBuffer:a  offset:0 atIndex:4];
    [enc setBuffer:b  offset:0 atIndex:5];
    [enc setBuffer:c  offset:0 atIndex:6];
    [enc setBuffer:d  offset:0 atIndex:7];
    [enc setBuffer:weightBuf offset:0 atIndex:8];
    [enc setBytes:&w      length:sizeof(int) atIndex:10];
    [enc setBytes:&h      length:sizeof(int) atIndex:11];
    [enc setBytes:&radius length:sizeof(int) atIndex:12];
    dispatch2D(enc, ps.gaussianBlurV4, w, h);
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
                    int p_PrematteEnabled, int p_PrematteBlur, int p_PrematteErode, int p_PrematteIterations,
                    int p_GuidedFilterEnabled, int p_GuidedFilterMode,
                    int p_GuidedRadius, float p_GuidedEpsilon,
                    float p_GuidedMix, float p_EdgeProtect, int p_RefineIterations,
                    float p_EdgeColorCorrect,
                    int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount,
                    int p_AdditiveKeyEnabled, int p_AdditiveKeyMode,
                    float p_AdditiveKeySat, float p_AdditiveKeyAmount, int p_AdditiveKeyBlackClamp,
                    int p_ViewMode,
                    const float* p_Input, const float* p_Screen,
                    const float* p_Background,
                    const float* p_GarbageMatte, const float* p_OcclusionMatte,
                    float* p_Output)
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
        // Keep Metal math conservative here. The private branch was using fast math to chase speed,
        // but this cross-platform port leans on CPU as the parity anchor. Safe math makes it much
        // easier to compare Metal against CPU/CUDA without backend-specific numeric drift muddying
        // whether a feature port is actually correct.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        options.mathMode = MTLMathModeSafe;
#else
        options.fastMathEnabled = NO;
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
        qs.pipes.gaussianBlurH4  = makePipeline(lib, "GaussianBlurH4");
        qs.pipes.gaussianBlurV4  = makePipeline(lib, "GaussianBlurV4");
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
        qs.pipes.rgbComputeProducts = makePipeline(lib, "RGBComputeProducts");
        qs.pipes.rgbGuidedCoeff     = makePipeline(lib, "RGBGuidedCoeff");
        qs.pipes.rgbGuidedEval      = makePipeline(lib, "RGBGuidedEval");
        qs.pipes.rgbGuidedApply     = makePipeline(lib, "RGBGuidedApply");
        qs.pipes.writeAlpha         = makePipeline(lib, "WriteAlphaKernel");
        qs.pipes.erodeAlpha         = makePipeline(lib, "ErodeAlphaKernel");
        qs.pipes.cleanPlateEstimate = makePipeline(lib, "CleanPlateEstimateKernel");
        qs.pipes.packRGBA           = makePipeline(lib, "PackRGBAKernel");
        qs.pipes.applyMatte         = makePipeline(lib, "ApplyMatteKernel");
        qs.pipes.diagnosticOutput   = makePipeline(lib, "DiagnosticOutputKernel");
        qs.pipes.gaussianBlurHTiled = makePipeline(lib, "GaussianBlurH_Tiled");
        qs.pipes.gaussianBlurVTiled = makePipeline(lib, "GaussianBlurV_Tiled");
        qs.pipes.additiveKey        = makePipeline(lib, "AdditiveKeyKernel");
        [lib release];
        s_QueueStateMap[queue] = qs;
        it = s_QueueStateMap.find(queue);
    }
    QueueState& state = it->second;
    PipelineSet& pipes = state.pipes;

    // ── Cached temp buffers (reused across frames) ──
    size_t chanBytes = (size_t)p_Width * (size_t)p_Height * sizeof(float);
    bool doPrematte = p_PrematteEnabled && p_PrematteBlur > 0;
    bool doGF = p_GuidedFilterEnabled && p_GuidedRadius > 0;
    bool doBgWrap = p_BgWrapEnabled && p_Background && p_BgWrapAmount > 0.0f;
    bool doAdditiveKey = p_AdditiveKeyEnabled && p_AdditiveKeyAmount > 0.0f;
    bool needBgBlur = doBgWrap || (doAdditiveKey && p_AdditiveKeyMode == 1 && p_Background != nullptr);
    bool rgbGF = doGF && (p_GuidedFilterMode == 1);

    // Determine how many temp buffers are needed
    int neededBufs = 2;  // always need temp[0..1] for core keyer output
    if (doPrematte)                      neededBufs = std::max(neededBufs, 7);  // prematte uses temp[0..6]
    if (rgbGF)                           neededBufs = 18;  // temp[0..17]
    else if (doGF || needBgBlur)         neededBufs = std::max(neededBufs, 7);  // temp[0..6]

    // Reallocate if resolution changed
    if (chanBytes != state.cachedChanBytes) {
        for (int i = 0; i < QueueState::MAX_TEMP; i++) {
            if (state.temp[i]) { [state.temp[i] release]; state.temp[i] = nil; }
        }
        if (state.cleanPlateBuf) { [state.cleanPlateBuf release]; state.cleanPlateBuf = nil; }
        state.cachedChanBytes = chanBytes;
        state.cachedCleanPlateBytes = 0;
        state.cachedBufCount = 0;
    }

    // Allocate temp buffers on demand
    if (neededBufs > state.cachedBufCount) {
        for (int i = state.cachedBufCount; i < neededBufs; i++) {
            if (!state.temp[i])
                state.temp[i] = [device newBufferWithLength:chanBytes options:MTLResourceStorageModePrivate];
        }
        state.cachedBufCount = neededBufs;
    }

    // Allocate RGBA clean plate buffer for prematte
    size_t rgbaBytes = chanBytes * 4;
    if (doPrematte && state.cachedCleanPlateBytes != rgbaBytes) {
        if (state.cleanPlateBuf) { [state.cleanPlateBuf release]; state.cleanPlateBuf = nil; }
        state.cleanPlateBuf = [device newBufferWithLength:rgbaBytes options:MTLResourceStorageModePrivate];
        state.cachedCleanPlateBytes = rgbaBytes;
    }

    // Scalar aliases (backward compatible with existing scalar dispatch code)
    id<MTLBuffer> tempA = state.temp[0];
    id<MTLBuffer> tempB = state.temp[1];
    id<MTLBuffer> tempC = (neededBufs > 2) ? state.temp[2] : nil;
    id<MTLBuffer> tempD = (neededBufs > 3) ? state.temp[3] : nil;
    id<MTLBuffer> tempE = (neededBufs > 4) ? state.temp[4] : nil;
    id<MTLBuffer> tempF = (neededBufs > 5) ? state.temp[5] : nil;
    id<MTLBuffer> tempG = (neededBufs > 6) ? state.temp[6] : nil;
    id<MTLBuffer>* t = state.temp;   // direct array access for RGB path
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

    // ── Cached Gaussian weights — only re-created when radius changes ──
    id<MTLBuffer> weightBuf = nil;
    if (doGF) {
        int r = p_GuidedRadius;
        if (state.gfWeightRadius != r) {
            if (state.gfWeightBuf) [state.gfWeightBuf release];
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
            state.gfWeightBuf = [device newBufferWithBytes:weights
                                                    length:kernelSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
            state.gfWeightRadius = r;
        }
        weightBuf = state.gfWeightBuf;
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

    // ══════════════════════════════════════════════════════════════════════
    //  PREMATTE — IBKColour-style synthetic clean plate generation
    //  1. Erode initial alpha (prevents FG bleed into plate estimate)
    //  2. Estimate clean plate: blend source → picked screen based on alpha
    //  3. Blur the plate (fills FG holes from surrounding BG information)
    //  4. Pack into RGBA and re-run core keyer with the clean plate as screen
    // ══════════════════════════════════════════════════════════════════════
    if (doPrematte) {
        // Cached Gaussian weights for prematte blur
        int pmR = std::max(1, p_PrematteBlur);
        if (state.pmWeightRadius != pmR) {
            if (state.pmWeightBuf) [state.pmWeightBuf release];
            int pmKernelSize = 2 * pmR + 1;
            float pmSigma = fmaxf(pmR / 3.0f, 0.5f);
            float pmInv2s2 = 1.0f / (2.0f * pmSigma * pmSigma);
            float* pmW = (float*)alloca(pmKernelSize * sizeof(float));
            float pmSum = 0.0f;
            for (int i = -pmR; i <= pmR; i++) {
                float wt = expf(-(float)(i * i) * pmInv2s2);
                pmW[i + pmR] = wt;
                pmSum += wt;
            }
            for (int i = 0; i < pmKernelSize; i++) pmW[i] /= pmSum;
            state.pmWeightBuf = [device newBufferWithBytes:pmW
                                                    length:pmKernelSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
            state.pmWeightRadius = pmR;
        }
        id<MTLBuffer> pmWeightBuf = state.pmWeightBuf;

        int pmIter = std::max(1, std::min(p_PrematteIterations, 5));
        for (int pi = 0; pi < pmIter; pi++) {

        // Step 1: Erode alpha (tempA → tempC, or use tempA directly if no erode)
        id<MTLBuffer> erodeAlpha = tempA;
        if (p_PrematteErode > 0) {
            [enc setComputePipelineState:pipes.erodeAlpha];
            [enc setBuffer:tempA offset:0 atIndex:0];
            [enc setBuffer:tempC offset:0 atIndex:1];
            [enc setBytes:&p_Width         length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height        length:sizeof(int) atIndex:11];
            [enc setBytes:&p_PrematteErode length:sizeof(int) atIndex:12];
            dispatch2D(enc, pipes.erodeAlpha, p_Width, p_Height);
            erodeAlpha = tempC;
        }

        // Step 2: Estimate clean plate (source + eroded alpha → 3 channel buffers)
        [enc setComputePipelineState:pipes.cleanPlateEstimate];
        [enc setBuffer:srcBuf    offset:0 atIndex:0];
        [enc setBuffer:erodeAlpha offset:0 atIndex:1];
        [enc setBuffer:tempD     offset:0 atIndex:2];   // cleanR
        [enc setBuffer:tempE     offset:0 atIndex:3];   // cleanG
        [enc setBuffer:tempF     offset:0 atIndex:4];   // cleanB
        [enc setBytes:&p_Width  length:sizeof(int)   atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int)   atIndex:11];
        [enc setBytes:&p_PickR  length:sizeof(float) atIndex:12];
        [enc setBytes:&p_PickG  length:sizeof(float) atIndex:13];
        [enc setBytes:&p_PickB  length:sizeof(float) atIndex:14];
        dispatch2D(enc, pipes.cleanPlateEstimate, p_Width, p_Height);

        // Step 3: Blur each channel of the clean plate
        // Use tempA as blur scratch (it's consumed; will be overwritten by re-key)
        gaussianBlur(enc, pipes, tempD, tempA, pmWeightBuf, p_Width, p_Height, pmR);
        gaussianBlur(enc, pipes, tempE, tempA, pmWeightBuf, p_Width, p_Height, pmR);
        gaussianBlur(enc, pipes, tempF, tempA, pmWeightBuf, p_Width, p_Height, pmR);

        // Step 4: Pack into RGBA clean plate buffer
        [enc setComputePipelineState:pipes.packRGBA];
        [enc setBuffer:tempD              offset:0 atIndex:0];   // R
        [enc setBuffer:tempE              offset:0 atIndex:1];   // G
        [enc setBuffer:tempF              offset:0 atIndex:2];   // B
        [enc setBuffer:state.cleanPlateBuf offset:0 atIndex:3];  // RGBA out
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        dispatch2D(enc, pipes.packRGBA, p_Width, p_Height);

        // Step 5: Re-run core keyer with clean plate as screen
        int useScreenOn = 1;
        [enc setComputePipelineState:pipes.coreKeyer];
        [enc setBuffer:srcBuf              offset:0 atIndex:0];
        [enc setBuffer:state.cleanPlateBuf offset:0 atIndex:1];   // clean plate as screen
        [enc setBuffer:dstBuf              offset:0 atIndex:2];
        [enc setBuffer:tempA               offset:0 atIndex:3];   // fresh raw alpha
        [enc setBuffer:tempB               offset:0 atIndex:4];   // fresh guide
        [enc setBytes:&p_Width           length:sizeof(int)   atIndex:10];
        [enc setBytes:&p_Height          length:sizeof(int)   atIndex:11];
        [enc setBytes:&p_ScreenColor     length:sizeof(int)   atIndex:12];
        [enc setBytes:&useScreenOn       length:sizeof(int)   atIndex:13];
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

        } // prematte iteration loop
    }

    // After prematte (or core if prematte off): dstBuf=RGBA, tempA=alpha, tempB=guide

    bool viewDone = false;

    // ── Diagnostic: Clean Plate ──────────────────────────────────────────
    if (p_ViewMode == 2) {
        if (doPrematte && state.cleanPlateBuf) {
            int dMode = 1;  // RGBA copy
            [enc setComputePipelineState:pipes.diagnosticOutput];
            [enc setBuffer:state.cleanPlateBuf offset:0 atIndex:0];
            [enc setBuffer:state.cleanPlateBuf offset:0 atIndex:1];
            [enc setBuffer:dstBuf              offset:0 atIndex:2];
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
            [enc setBytes:&dMode    length:sizeof(int) atIndex:12];
            dispatch2D(enc, pipes.diagnosticOutput, p_Width, p_Height);
        } else {
            // Prematte off: show the screen input as fallback
            int dMode = 1;
            [enc setComputePipelineState:pipes.diagnosticOutput];
            [enc setBuffer:scrBuf offset:0 atIndex:0];
            [enc setBuffer:scrBuf offset:0 atIndex:1];
            [enc setBuffer:dstBuf offset:0 atIndex:2];
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
            [enc setBytes:&dMode    length:sizeof(int) atIndex:12];
            dispatch2D(enc, pipes.diagnosticOutput, p_Width, p_Height);
        }
        viewDone = true;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  EXTERNAL MATTES — Garbage and Occlusion
    //  Applied after core keyer (+ prematte) but before guided filter,
    //  so the GF refines the combined matte with external constraints.
    // ══════════════════════════════════════════════════════════════════════
    if (p_GarbageMatte && !viewDone) {
        id<MTLBuffer> garbageBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_GarbageMatte));
        int mode = 0;
        [enc setComputePipelineState:pipes.applyMatte];
        [enc setBuffer:tempA      offset:0 atIndex:0];
        [enc setBuffer:dstBuf     offset:0 atIndex:1];
        [enc setBuffer:garbageBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        [enc setBytes:&mode     length:sizeof(int) atIndex:12];
        dispatch2D(enc, pipes.applyMatte, p_Width, p_Height);
    }

    if (p_OcclusionMatte && !viewDone) {
        id<MTLBuffer> occlusionBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_OcclusionMatte));
        int mode = 1;
        [enc setComputePipelineState:pipes.applyMatte];
        [enc setBuffer:tempA        offset:0 atIndex:0];
        [enc setBuffer:dstBuf       offset:0 atIndex:1];
        [enc setBuffer:occlusionBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        [enc setBytes:&mode     length:sizeof(int) atIndex:12];
        dispatch2D(enc, pipes.applyMatte, p_Width, p_Height);
    }

    // ── Diagnostic: Raw Matte ────────────────────────────────────────────
    if (p_ViewMode == 1 && !viewDone) {
        int dMode = 0;  // 1ch alpha → greyscale
        [enc setComputePipelineState:pipes.diagnosticOutput];
        [enc setBuffer:tempA  offset:0 atIndex:0];
        [enc setBuffer:tempA  offset:0 atIndex:1];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        [enc setBytes:&dMode    length:sizeof(int) atIndex:12];
        dispatch2D(enc, pipes.diagnosticOutput, p_Width, p_Height);
        viewDone = true;
    }

    if (!viewDone && doGF && !rgbGF) {
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

    } else if (!viewDone && rgbGF) {
        // ══════════════════════════════════════════════════════════════════
        //  RGB COLOR-AWARE GUIDED FILTER
        //  Uses full 3-channel RGB guide with 3×3 covariance matrix.
        //  Dramatically better color-edge awareness than scalar luminance.
        //  Optimized: 4-channel vectorized blur reduces dispatches 3.4×.
        // ══════════════════════════════════════════════════════════════════
        int r = p_GuidedRadius;
        int numIter = std::max(1, std::min(p_RefineIterations, 5));

        // Save raw alpha to t[17] for final mix
        [enc setComputePipelineState:pipes.copyBuffer];
        [enc setBuffer:t[0]  offset:0 atIndex:3];    // raw alpha from core keyer
        [enc setBuffer:t[17] offset:0 atIndex:4];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        dispatch2D(enc, pipes.copyBuffer, p_Width, p_Height);

        for (int iter = 0; iter < numIter; iter++) {
            bool isLast = (iter == numIter - 1);

            // ── Compute products: source RGB + current alpha → 13 channels ──
            [enc setComputePipelineState:pipes.rgbComputeProducts];
            [enc setBuffer:srcBuf offset:0 atIndex:0];   // source RGBA (guide RGB)
            [enc setBuffer:dstBuf offset:0 atIndex:1];   // output (alpha in .w)
            [enc setBuffer:t[0]   offset:0 atIndex:2];   // → mean_Ir
            [enc setBuffer:t[1]   offset:0 atIndex:3];   // → mean_Ig
            [enc setBuffer:t[2]   offset:0 atIndex:4];   // → mean_Ib
            [enc setBuffer:t[3]   offset:0 atIndex:5];   // → mean_p
            [enc setBuffer:t[4]   offset:0 atIndex:6];   // → IrIr
            [enc setBuffer:t[5]   offset:0 atIndex:7];   // → IrIg
            [enc setBuffer:t[6]   offset:0 atIndex:8];   // → IrIb
            [enc setBuffer:t[7]   offset:0 atIndex:9];   // → IgIg
            [enc setBuffer:t[8]   offset:0 atIndex:10];  // → IgIb
            [enc setBuffer:t[9]   offset:0 atIndex:11];  // → IbIb
            [enc setBuffer:t[10]  offset:0 atIndex:12];  // → Irp
            [enc setBuffer:t[11]  offset:0 atIndex:13];  // → Igp
            [enc setBuffer:t[12]  offset:0 atIndex:14];  // → Ibp
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:20];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:21];
            dispatch2D(enc, pipes.rgbComputeProducts, p_Width, p_Height);

            // ── Blur 13 channels: 3 groups of 4 + 1 single ──
            // (t[13..16] are dedicated scratch for blur4)
            gaussianBlur4(enc, pipes,
                          t[0], t[1], t[2], t[3],
                          t[13], t[14], t[15], t[16],
                          weightBuf, p_Width, p_Height, r);
            gaussianBlur4(enc, pipes,
                          t[4], t[5], t[6], t[7],
                          t[13], t[14], t[15], t[16],
                          weightBuf, p_Width, p_Height, r);
            gaussianBlur4(enc, pipes,
                          t[8], t[9], t[10], t[11],
                          t[13], t[14], t[15], t[16],
                          weightBuf, p_Width, p_Height, r);
            gaussianBlur(enc, pipes, t[12], t[13], weightBuf, p_Width, p_Height, r);

            // ── Compute 3×3 coefficients: ar, ag, ab, b ──
            // Writes: t[4]=ar, t[7]=ag, t[9]=ab, t[0]=b
            // (b goes to t[0] since means are consumed; t[13..16] stay as scratch)
            [enc setComputePipelineState:pipes.rgbGuidedCoeff];
            [enc setBuffer:t[0]   offset:0 atIndex:0];   // mean_Ir (consumed → becomes b output)
            [enc setBuffer:t[1]   offset:0 atIndex:1];   // mean_Ig
            [enc setBuffer:t[2]   offset:0 atIndex:2];   // mean_Ib
            [enc setBuffer:t[3]   offset:0 atIndex:3];   // mean_p
            [enc setBuffer:t[4]   offset:0 atIndex:4];   // IrIr → ar
            [enc setBuffer:t[5]   offset:0 atIndex:5];   // IrIg
            [enc setBuffer:t[6]   offset:0 atIndex:6];   // IrIb
            [enc setBuffer:t[7]   offset:0 atIndex:7];   // IgIg → ag
            [enc setBuffer:t[8]   offset:0 atIndex:8];   // IgIb
            [enc setBuffer:t[9]   offset:0 atIndex:9];   // IbIb → ab
            [enc setBuffer:t[10]  offset:0 atIndex:10];  // Irp
            [enc setBuffer:t[11]  offset:0 atIndex:11];  // Igp
            [enc setBuffer:t[12]  offset:0 atIndex:12];  // Ibp
            [enc setBuffer:t[0]   offset:0 atIndex:13];  // → b (overwrites mean_Ir)
            [enc setBytes:&p_Width          length:sizeof(int)   atIndex:20];
            [enc setBytes:&p_Height         length:sizeof(int)   atIndex:21];
            [enc setBytes:&p_GuidedEpsilon  length:sizeof(float) atIndex:22];
            dispatch2D(enc, pipes.rgbGuidedCoeff, p_Width, p_Height);

            // After coeff: t[4]=ar, t[7]=ag, t[9]=ab, t[0]=b
            // Blur 4 coefficients in a single vectorized pass
            gaussianBlur4(enc, pipes,
                          t[4], t[7], t[9], t[0],
                          t[13], t[14], t[15], t[16],
                          weightBuf, p_Width, p_Height, r);

            if (isLast) {
                // Final: apply with mix against saved raw alpha + premultiply
                [enc setComputePipelineState:pipes.rgbGuidedApply];
                [enc setBuffer:srcBuf offset:0 atIndex:0];   // source RGB
                [enc setBuffer:dstBuf offset:0 atIndex:1];   // output RGBA
                [enc setBuffer:t[4]   offset:0 atIndex:2];   // mean_ar
                [enc setBuffer:t[7]   offset:0 atIndex:3];   // mean_ag
                [enc setBuffer:t[9]   offset:0 atIndex:4];   // mean_ab
                [enc setBuffer:t[0]   offset:0 atIndex:5];   // mean_b
                [enc setBuffer:t[17]  offset:0 atIndex:6];   // saved raw alpha
                [enc setBytes:&p_Width        length:sizeof(int)   atIndex:20];
                [enc setBytes:&p_Height       length:sizeof(int)   atIndex:21];
                [enc setBytes:&p_Premultiply  length:sizeof(int)   atIndex:22];
                [enc setBytes:&p_GuidedMix    length:sizeof(float) atIndex:23];
                dispatch2D(enc, pipes.rgbGuidedApply, p_Width, p_Height);
            } else {
                // Intermediate: eval refined alpha → t[1], then write to dstBuf alpha
                [enc setComputePipelineState:pipes.rgbGuidedEval];
                [enc setBuffer:srcBuf offset:0 atIndex:0];   // source RGB
                [enc setBuffer:t[4]   offset:0 atIndex:2];   // mean_ar
                [enc setBuffer:t[7]   offset:0 atIndex:3];   // mean_ag
                [enc setBuffer:t[9]   offset:0 atIndex:4];   // mean_ab
                [enc setBuffer:t[0]   offset:0 atIndex:5];   // mean_b
                [enc setBuffer:t[1]   offset:0 atIndex:6];   // output: refined alpha
                [enc setBytes:&p_Width  length:sizeof(int) atIndex:20];
                [enc setBytes:&p_Height length:sizeof(int) atIndex:21];
                dispatch2D(enc, pipes.rgbGuidedEval, p_Width, p_Height);

                // Write refined alpha back to dstBuf's .w channel for next iteration
                [enc setComputePipelineState:pipes.writeAlpha];
                [enc setBuffer:t[1]   offset:0 atIndex:0];   // 1ch refined alpha
                [enc setBuffer:dstBuf offset:0 atIndex:1];   // RGBA buffer
                [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
                [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
                dispatch2D(enc, pipes.writeAlpha, p_Width, p_Height);
            }
        } // RGB iteration loop

    } else if (!viewDone && p_Premultiply) {
        // No GF — just premultiply
        [enc setComputePipelineState:pipes.premultiply];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        dispatch2D(enc, pipes.premultiply, p_Width, p_Height);
    }

    // ── Diagnostic: Refined Matte ────────────────────────────────────────
    if (p_ViewMode == 3 && !viewDone) {
        int dMode = 2;  // extract alpha from RGBA → greyscale
        [enc setComputePipelineState:pipes.diagnosticOutput];
        [enc setBuffer:dstBuf offset:0 atIndex:0];
        [enc setBuffer:dstBuf offset:0 atIndex:1];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
        [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
        [enc setBytes:&dMode    length:sizeof(int) atIndex:12];
        dispatch2D(enc, pipes.diagnosticOutput, p_Width, p_Height);
        viewDone = true;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PASS 2.5: Edge Color Correction
    //  Re-estimates FG color at semi-transparent edges using the matting
    //  equation: fg = (src - screen*(1-alpha)) / alpha
    // ══════════════════════════════════════════════════════════════════════
    if (!viewDone && p_EdgeColorCorrect > 0.0f) {
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

    // ── Diagnostic: Despilled Source ─────────────────────────────────────
    if (p_ViewMode == 4 && !viewDone) {
        viewDone = true;  // skip BG wrap — output despilled FG as-is
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PASS 3: Background Stage — BG Wrap + Additive Key
    //  Blurs the BG and bleeds it into FG edges weighted by (1-alpha).
    //  Additive key: recovers fine detail the alpha missed by
    //  superimposing source-minus-screen onto the composite.
    // ══════════════════════════════════════════════════════════════════════

    // BG extraction/blur needed for: bg wrap OR additive key multiplication mode
    if (needBgBlur && !viewDone) {
        id<MTLBuffer> bgBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Background));

        // Cached Gaussian weights for BG blur
        int bwR = std::max(1, p_BgWrapBlur);
        if (state.bwWeightRadius != bwR) {
            if (state.bwWeightBuf) [state.bwWeightBuf release];
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
            state.bwWeightBuf = [device newBufferWithBytes:bwW
                                                    length:bwKernelSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
            state.bwWeightRadius = bwR;
        }
        id<MTLBuffer> bwWeightBuf = state.bwWeightBuf;

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

        if (p_ViewMode == 5) {
            // ── Diagnostic: Blurred Background ──────────────────────────
            [enc setComputePipelineState:pipes.packRGBA];
            [enc setBuffer:tempA  offset:0 atIndex:0];
            [enc setBuffer:tempB  offset:0 atIndex:1];
            [enc setBuffer:tempC  offset:0 atIndex:2];
            [enc setBuffer:dstBuf offset:0 atIndex:3];
            [enc setBytes:&p_Width  length:sizeof(int) atIndex:10];
            [enc setBytes:&p_Height length:sizeof(int) atIndex:11];
            dispatch2D(enc, pipes.packRGBA, p_Width, p_Height);
        } else {
            // Apply BG wrap (light wrap)
            if (doBgWrap) {
                [enc setComputePipelineState:pipes.bgWrap];
                [enc setBuffer:dstBuf offset:0 atIndex:2];
                [enc setBuffer:tempA  offset:0 atIndex:3];
                [enc setBuffer:tempB  offset:0 atIndex:4];
                [enc setBuffer:tempC  offset:0 atIndex:5];
                [enc setBytes:&p_Width       length:sizeof(int)   atIndex:10];
                [enc setBytes:&p_Height      length:sizeof(int)   atIndex:11];
                [enc setBytes:&p_BgWrapAmount length:sizeof(float) atIndex:12];
                dispatch2D(enc, pipes.bgWrap, p_Width, p_Height);
            }

            // Apply additive key (multiplication mode uses blurred BG in tempA/B/C)
            if (doAdditiveKey) {
                [enc setComputePipelineState:pipes.additiveKey];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:scrBuf offset:0 atIndex:1];
                [enc setBuffer:dstBuf offset:0 atIndex:2];
                [enc setBuffer:tempA  offset:0 atIndex:3];
                [enc setBuffer:tempB  offset:0 atIndex:4];
                [enc setBuffer:tempC  offset:0 atIndex:5];
                [enc setBytes:&p_Width              length:sizeof(int)   atIndex:10];
                [enc setBytes:&p_Height             length:sizeof(int)   atIndex:11];
                [enc setBytes:&p_AdditiveKeyMode    length:sizeof(int)   atIndex:12];
                [enc setBytes:&p_UseScreenInput     length:sizeof(int)   atIndex:13];
                [enc setBytes:&p_PickR              length:sizeof(float) atIndex:14];
                [enc setBytes:&p_PickG              length:sizeof(float) atIndex:15];
                [enc setBytes:&p_PickB              length:sizeof(float) atIndex:16];
                [enc setBytes:&p_AdditiveKeySat     length:sizeof(float) atIndex:17];
                [enc setBytes:&p_AdditiveKeyAmount  length:sizeof(float) atIndex:18];
                [enc setBytes:&p_AdditiveKeyBlackClamp length:sizeof(int) atIndex:19];
                dispatch2D(enc, pipes.additiveKey, p_Width, p_Height);
            }
        }
    }

    // Additive key (addition mode) — works without BG input
    if (doAdditiveKey && p_AdditiveKeyMode == 0 && !needBgBlur && !viewDone) {
        [enc setComputePipelineState:pipes.additiveKey];
        [enc setBuffer:srcBuf offset:0 atIndex:0];
        [enc setBuffer:scrBuf offset:0 atIndex:1];
        [enc setBuffer:dstBuf offset:0 atIndex:2];
        [enc setBuffer:tempA  offset:0 atIndex:3];  // unused in addition mode
        [enc setBuffer:tempB  offset:0 atIndex:4];
        [enc setBuffer:tempC  offset:0 atIndex:5];
        [enc setBytes:&p_Width              length:sizeof(int)   atIndex:10];
        [enc setBytes:&p_Height             length:sizeof(int)   atIndex:11];
        [enc setBytes:&p_AdditiveKeyMode    length:sizeof(int)   atIndex:12];
        [enc setBytes:&p_UseScreenInput     length:sizeof(int)   atIndex:13];
        [enc setBytes:&p_PickR              length:sizeof(float) atIndex:14];
        [enc setBytes:&p_PickG              length:sizeof(float) atIndex:15];
        [enc setBytes:&p_PickB              length:sizeof(float) atIndex:16];
        [enc setBytes:&p_AdditiveKeySat     length:sizeof(float) atIndex:17];
        [enc setBytes:&p_AdditiveKeyAmount  length:sizeof(float) atIndex:18];
        [enc setBytes:&p_AdditiveKeyBlackClamp length:sizeof(int) atIndex:19];
        dispatch2D(enc, pipes.additiveKey, p_Width, p_Height);
    }

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // ── Cleanup (temp + weight buffers are cached — only release per-frame objects) ──
    if (createdDummy) [scrBuf release];
  } // @autoreleasepool
}
