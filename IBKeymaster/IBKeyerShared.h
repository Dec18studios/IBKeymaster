#pragma once

#include <algorithm>
#include <cmath>

#if defined(__CUDACC__)
#define IBKEYER_HOST_DEVICE __host__ __device__
#else
#define IBKEYER_HOST_DEVICE
#endif

namespace IBKeyerCore {

enum ScreenColor
{
    kScreenRed = 0,
    kScreenGreen = 1,
    kScreenBlue = 2
};

struct IBKeyerParams
{
    int screenColor = kScreenBlue;
    bool useScreenInput = true;
    float pickR = 0.0f;
    float pickG = 0.0f;
    float pickB = 0.0f;
    float bias = 0.5f;
    float limit = 1.0f;
    float respillR = 0.0f;
    float respillG = 0.0f;
    float respillB = 0.0f;
    bool premultiply = false;
    bool nearGreyExtract = true;
    float nearGreyAmount = 1.0f;
    float blackClip = 0.0f;
    float whiteClip = 1.0f;
    bool guidedFilterEnabled = true;
    int guidedRadius = 8;
    float guidedEpsilon = 0.01f;
    float guidedMix = 1.0f;
};

struct PackedFrame
{
    int width = 0;
    int height = 0;
    const float* srcRgba = nullptr;
    const float* screenRgba = nullptr;
    float* dstRgba = nullptr;
};

IBKEYER_HOST_DEVICE inline float clamp01(float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

IBKEYER_HOST_DEVICE inline float safeDivide(float numerator, float denominator)
{
    return (fabsf(denominator) > 1e-8f) ? numerator / denominator : 0.0f;
}

IBKEYER_HOST_DEVICE inline void reorderChannels(float r, float g, float b, int screenColor,
                                                float& c0, float& c1, float& c2)
{
    if (screenColor == kScreenRed) {
        c0 = r;
        c1 = g;
        c2 = b;
    } else if (screenColor == kScreenGreen) {
        c0 = g;
        c1 = r;
        c2 = b;
    } else {
        c0 = b;
        c1 = r;
        c2 = g;
    }
}

IBKEYER_HOST_DEVICE inline float despillValue(float r, float g, float b, int screenColor,
                                              float bias, float limit)
{
    float c0, c1, c2;
    reorderChannels(r, g, b, screenColor, c0, c1, c2);
    return c0 - (c1 * bias + c2 * (1.0f - bias)) * limit;
}

IBKEYER_HOST_DEVICE inline float nearGreyAlpha(float r, float g, float b, int screenColor,
                                               float amount)
{
    float c0, c1, c2;
    reorderChannels(r, g, b, screenColor, c0, c1, c2);
    const float mx = fmaxf(c0, fmaxf(c1, c2));
    const float comp = (mx == c1) ? c1 : c2;
    const float value = c0 * (1.0f - amount) + comp * amount;
    return clamp01(value);
}

IBKEYER_HOST_DEVICE inline float luminance(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

inline bool guidedFilterActive(const IBKeyerParams& params)
{
    return params.guidedFilterEnabled && params.guidedRadius > 0;
}

} // namespace IBKeyerCore
