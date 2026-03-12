/////// IBKeymaster v2 OFX Plugin
// Guided-Filter Enhanced Image-Based Keyer
// Port of Jed Smith's IBKeymaster + guided filter matte refinement + matte controls
#include "IBKeymaster.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <memory>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
#define kPluginName "IBKeymaster"
#define kPluginGrouping "create@Dec18Studios.com"
#define kPluginDescription \
    "Image-Based Keyer with Guided Filter refinement.\n\n" \
    "Extracts a high-quality matte and despilled foreground from green/blue screen footage " \
    "by comparing source pixels against a clean screen plate or pick colour.\n\n" \
    "The Guided Filter uses the source luminance as an edge-aware guide to refine " \
    "the raw colour-difference matte, recovering hair detail, transparency, " \
    "and motion blur that traditional per-pixel keyers lose.\n\n" \
    "Based on IBKeymaster by Jed Smith (gaffer-tools) + He et al. guided filter."
#define kPluginIdentifier "com.OpenFXSample.IBKeymaster"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

// Screen colour enum
#define kScreenRed 0
#define kScreenGreen 1
#define kScreenBlue 2

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR CLASS
////////////////////////////////////////////////////////////////////////////////

class ImageProcessor : public OFX::ImageProcessor
{
public:
    explicit ImageProcessor(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void processImagesMetal();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScreenImg(OFX::Image* p_ScreenImg);
    void setBgImg(OFX::Image* p_BgImg);
    void setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4);

    void setKeyerParams(int p_ScreenColor, int p_UseScreenInput,
                        float p_PickR, float p_PickG, float p_PickB,
                        float p_Bias, float p_Limit,
                        float p_RespillR, float p_RespillG, float p_RespillB,
                        int p_Premultiply, int p_NearGreyExtract,
                        float p_NearGreyAmount, float p_NearGreySoftness,
                        float p_BlackClip, float p_WhiteClip, float p_MatteGamma,
                        int p_GuidedFilterEnabled, int p_GuidedRadius,
                        float p_GuidedEpsilon, float p_GuidedMix,
                        float p_EdgeProtect, int p_RefineIterations, float p_EdgeColorCorrect,
                        int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount);

private:
    OFX::Image* _srcImg;
    OFX::Image* _screenImg;
    OFX::Image* _bgImg;
    float _scales[4];

    // Core keyer params
    int   _screenColor;
    int   _useScreenInput;
    float _pickR, _pickG, _pickB;
    float _bias;
    float _limit;
    float _respillR, _respillG, _respillB;
    int   _premultiply;
    int   _nearGreyExtract;
    float _nearGreyAmount;
    float _nearGreySoftness;

    // Matte controls
    float _blackClip;
    float _whiteClip;
    float _matteGamma;

    // Guided filter params
    int   _guidedFilterEnabled;
    int   _guidedRadius;
    float _guidedEpsilon;
    float _guidedMix;
    float _edgeProtect;
    int   _refineIterations;
    float _edgeColorCorrect;

    // Background wrap params
    int   _bgWrapEnabled;
    int   _bgWrapBlur;
    float _bgWrapAmount;

    // Helpers
    inline void reorder(float r, float g, float b,
                        float& c0, float& c1, float& c2) const
    {
        switch (_screenColor) {
            case kScreenRed:   c0 = r; c1 = g; c2 = b; break;
            case kScreenGreen: c0 = g; c1 = r; c2 = b; break;
            case kScreenBlue:
            default:           c0 = b; c1 = r; c2 = g; break;
        }
    }

    inline float despill(float r, float g, float b) const
    {
        float c0, c1, c2;
        reorder(r, g, b, c0, c1, c2);
        return c0 - (c1 * _bias + c2 * (1.0f - _bias)) * _limit;
    }

    inline float despillNGE(float r, float g, float b) const
    {
        float c0, c1, c2;
        reorder(r, g, b, c0, c1, c2);
        float s0 = _nearGreySoftness;
        float mx = fmaxf(c0, fmaxf(c1, c2));
        float comp = (mx == c1) ? c1 : c2;
        float val = c0 * (1.0f - s0) + comp * s0;
        return fmaxf(0.0f, fminf(1.0f, val));
    }

    static inline float safeDivide(float a, float b)
    {
        return (fabsf(b) > 1e-8f) ? a / b : 0.0f;
    }

    static inline float luminance(float r, float g, float b)
    {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

ImageProcessor::ImageProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

void ImageProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageProcessor::setScreenImg(OFX::Image* p_ScreenImg)
{
    _screenImg = p_ScreenImg;
}

void ImageProcessor::setBgImg(OFX::Image* p_BgImg)
{
    _bgImg = p_BgImg;
}

void ImageProcessor::setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4)
{
    _scales[0] = p_Scale1;
    _scales[1] = p_Scale2;
    _scales[2] = p_Scale3;
    _scales[3] = p_Scale4;
}

void ImageProcessor::setKeyerParams(int p_ScreenColor, int p_UseScreenInput,
                                    float p_PickR, float p_PickG, float p_PickB,
                                    float p_Bias, float p_Limit,
                                    float p_RespillR, float p_RespillG, float p_RespillB,
                                    int p_Premultiply, int p_NearGreyExtract,
                                    float p_NearGreyAmount, float p_NearGreySoftness,
                                    float p_BlackClip, float p_WhiteClip, float p_MatteGamma,
                                    int p_GuidedFilterEnabled, int p_GuidedRadius,
                                    float p_GuidedEpsilon, float p_GuidedMix,
                                    float p_EdgeProtect, int p_RefineIterations, float p_EdgeColorCorrect,
                                    int p_BgWrapEnabled, int p_BgWrapBlur, float p_BgWrapAmount)
{
    _screenColor     = p_ScreenColor;
    _useScreenInput  = p_UseScreenInput;
    _pickR = p_PickR; _pickG = p_PickG; _pickB = p_PickB;
    _bias            = p_Bias;
    _limit           = p_Limit;
    _respillR = p_RespillR; _respillG = p_RespillG; _respillB = p_RespillB;
    _premultiply     = p_Premultiply;
    _nearGreyExtract = p_NearGreyExtract;
    _nearGreyAmount  = p_NearGreyAmount;
    _nearGreySoftness = p_NearGreySoftness;
    _blackClip       = p_BlackClip;
    _whiteClip       = p_WhiteClip;
    _matteGamma      = p_MatteGamma;
    _guidedFilterEnabled = p_GuidedFilterEnabled;
    _guidedRadius    = p_GuidedRadius;
    _guidedEpsilon   = p_GuidedEpsilon;
    _guidedMix       = p_GuidedMix;
    _edgeProtect     = p_EdgeProtect;
    _refineIterations = p_RefineIterations;
    _edgeColorCorrect = p_EdgeColorCorrect;
    _bgWrapEnabled   = p_BgWrapEnabled;
    _bgWrapBlur      = p_BgWrapBlur;
    _bgWrapAmount    = p_BgWrapAmount;
}

////////////////////////////////////////////////////////////////////////////////
// GPU PROCESSING — EXTERNAL KERNEL DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height,
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

#ifdef __APPLE__
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

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
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

////////////////////////////////////////////////////////////////////////////////
// GPU DISPATCH METHODS
////////////////////////////////////////////////////////////////////////////////

void ImageProcessor::processImagesCUDA()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    float* screen = (_screenImg && _useScreenInput) ? static_cast<float*>(_screenImg->getPixelData()) : nullptr;
    float* background = (_bgImg && _bgWrapEnabled) ? static_cast<float*>(_bgImg->getPixelData()) : nullptr;

    RunCudaKernel(_pCudaStream, width, height,
                  _screenColor, _useScreenInput,
                  _pickR, _pickG, _pickB,
                  _bias, _limit,
                  _respillR, _respillG, _respillB,
                  _premultiply, _nearGreyExtract, _nearGreyAmount, _nearGreySoftness,
                  _blackClip, _whiteClip, _matteGamma,
                  _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                  _edgeProtect, _refineIterations, _edgeColorCorrect,
                  _bgWrapEnabled, _bgWrapBlur, _bgWrapAmount,
                  input, screen, background, output);
#endif
}

void ImageProcessor::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    float* screen = (_screenImg && _useScreenInput) ? static_cast<float*>(_screenImg->getPixelData()) : nullptr;
    float* background = (_bgImg && _bgWrapEnabled) ? static_cast<float*>(_bgImg->getPixelData()) : nullptr;

    RunMetalKernel(_pMetalCmdQ, width, height,
                   _screenColor, _useScreenInput,
                   _pickR, _pickG, _pickB,
                   _bias, _limit,
                   _respillR, _respillG, _respillB,
                   _premultiply, _nearGreyExtract, _nearGreyAmount, _nearGreySoftness,
                   _blackClip, _whiteClip, _matteGamma,
                   _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                   _edgeProtect, _refineIterations, _edgeColorCorrect,
                   _bgWrapEnabled, _bgWrapBlur, _bgWrapAmount,
                   input, screen, background, output);
#endif
}

void ImageProcessor::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    float* screen = (_screenImg && _useScreenInput) ? static_cast<float*>(_screenImg->getPixelData()) : nullptr;
    float* background = (_bgImg && _bgWrapEnabled) ? static_cast<float*>(_bgImg->getPixelData()) : nullptr;

    RunOpenCLKernel(_pOpenCLCmdQ, width, height,
                    _screenColor, _useScreenInput,
                    _pickR, _pickG, _pickB,
                    _bias, _limit,
                    _respillR, _respillG, _respillB,
                    _premultiply, _nearGreyExtract, _nearGreyAmount, _nearGreySoftness,
                    _blackClip, _whiteClip, _matteGamma,
                    _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                    _edgeProtect, _refineIterations, _edgeColorCorrect,
                    _bgWrapEnabled, _bgWrapBlur, _bgWrapAmount,
                    input, screen, background, output);
}

////////////////////////////////////////////////////////////////////////////////
// CPU PROCESSING — FALLBACK (includes guided filter on CPU)
////////////////////////////////////////////////////////////////////////////////

// CPU Gaussian blur helper (single channel, in-place via scratch buffer, pre-computed weights)
static void cpuGaussianBlur(float* data, float* scratch, const float* weights,
                            int w, int h, int radius)
{
    // Horizontal pass: data → scratch
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int dx = -radius; dx <= radius; dx++) {
                int sx = std::max(0, std::min(w - 1, x + dx));
                sum += data[y * w + sx] * weights[dx + radius];
            }
            scratch[y * w + x] = sum;
        }
    }
    // Vertical pass: scratch → data
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int dy = -radius; dy <= radius; dy++) {
                int sy = std::max(0, std::min(h - 1, y + dy));
                sum += scratch[sy * w + x] * weights[dy + radius];
            }
            data[y * w + x] = sum;
        }
    }
}

void ImageProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    const int w = p_ProcWindow.x2 - p_ProcWindow.x1;
    const int h = p_ProcWindow.y2 - p_ProcWindow.y1;
    const int numPix = w * h;

    // Allocate temp arrays for guided filter
    float* rawAlphaArr = nullptr;
    float* guideArr = nullptr;
    float* meanI = nullptr;
    float* meanP = nullptr;
    float* meanIp = nullptr;
    float* meanII = nullptr;
    float* scratch = nullptr;

    bool doGF = _guidedFilterEnabled && _guidedRadius > 0;
    if (doGF) {
        rawAlphaArr = new float[numPix];
        guideArr    = new float[numPix];
        meanI       = new float[numPix];
        meanP       = new float[numPix];
        meanIp      = new float[numPix];
        meanII      = new float[numPix];
        scratch     = new float[numPix];
    }

    // ── PASS 1: Core keyer (per-pixel) ──────────────────────────────────
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            if (srcPix)
            {
                float srcR = srcPix[0];
                float srcG = srcPix[1];
                float srcB = srcPix[2];

                float scrR, scrG, scrB;
                if (_useScreenInput && _screenImg)
                {
                    float* scrPix = static_cast<float*>(_screenImg->getPixelAddress(x, y));
                    if (scrPix) { scrR = scrPix[0]; scrG = scrPix[1]; scrB = scrPix[2]; }
                    else { scrR = _pickR; scrG = _pickG; scrB = _pickB; }
                }
                else
                {
                    scrR = _pickR; scrG = _pickG; scrB = _pickB;
                }

                float despillRGB    = despill(srcR, srcG, srcB);
                float despillScreen = despill(scrR, scrG, scrB);

                float normalized = safeDivide(despillRGB, despillScreen);

                float spillMul = fmaxf(0.0f, normalized);
                float ssR = srcR - spillMul * scrR;
                float ssG = srcG - spillMul * scrG;
                float ssB = srcB - spillMul * scrB;

                float alpha = fmaxf(0.0f, fminf(1.0f, 1.0f - normalized));

                if (_nearGreyExtract && _nearGreyAmount > 0.0f)
                {
                    float divR = safeDivide(ssR, srcR);
                    float divG = safeDivide(ssG, srcG);
                    float divB = safeDivide(ssB, srcB);
                    float ngeAlpha = despillNGE(divR, divG, divB);
                    // Strength-blended screen composite:
                    // full screen = ngeA + alpha - ngeA * alpha = alpha + ngeA*(1-alpha)
                    // blend by amount: alpha + amount * ngeA * (1 - alpha)
                    alpha = alpha + _nearGreyAmount * ngeAlpha * (1.0f - alpha);
                }

                // Black/White clip
                float lo = _blackClip;
                float hi = _whiteClip;
                if (hi > lo + 1e-6f) {
                    alpha = fmaxf(0.0f, fminf(1.0f, (alpha - lo) / (hi - lo)));
                }

                // Matte Gamma — shape alpha falloff (helps motion blur)
                if (_matteGamma != 1.0f && alpha > 0.0f && alpha < 1.0f) {
                    alpha = powf(alpha, _matteGamma);
                }

                float respillMul = fmaxf(0.0f, despillScreen * normalized);
                float outR = ssR + respillMul * _respillR;
                float outG = ssG + respillMul * _respillG;
                float outB = ssB + respillMul * _respillB;

                dstPix[0] = outR;
                dstPix[1] = outG;
                dstPix[2] = outB;
                dstPix[3] = alpha;

                // Store for guided filter
                if (doGF) {
                    int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                    rawAlphaArr[li] = alpha;
                    float lum = luminance(srcR, srcG, srcB);
                    guideArr[li] = lum * (1.0f - _edgeProtect) + alpha * _edgeProtect;
                }
            }
            else
            {
                dstPix[0] = 0; dstPix[1] = 0; dstPix[2] = 0; dstPix[3] = 0;
                if (doGF) {
                    int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                    rawAlphaArr[li] = 0.0f;
                    guideArr[li] = 0.0f;
                }
            }
            dstPix += 4;
        }
    }

    // ── PASS 2: Guided filter (CPU, iterative refinement) ────────────────
    if (doGF && !_effect.abort())
    {
        int r = _guidedRadius;
        float eps = _guidedEpsilon;
        int numIter = std::max(1, std::min(_refineIterations, 5));

        // Pre-compute Gaussian weights
        int kernelSize = 2 * r + 1;
        float sigma = fmaxf(r / 3.0f, 0.5f);
        float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);
        float* gWeights = new float[kernelSize];
        float wsum = 0.0f;
        for (int i = -r; i <= r; i++) {
            float wt = expf(-(float)(i * i) * invTwoSigmaSq);
            gWeights[i + r] = wt;
            wsum += wt;
        }
        for (int i = 0; i < kernelSize; i++) gWeights[i] /= wsum;

        // Save original raw alpha for final mix
        float* savedRawAlpha = new float[numPix];
        for (int i = 0; i < numPix; i++) savedRawAlpha[i] = rawAlphaArr[i];

        // currentP starts as raw alpha
        // guideArr holds the current guide (updated each iteration)
        for (int iter = 0; iter < numIter; iter++) {
            if (_effect.abort()) break;

            if (iter > 0) {
                // Refine guide: use refined alpha + source RGB for better FG estimate
                for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
                    float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(p_ProcWindow.x1, y) : 0);
                    for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
                        int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                        float a = rawAlphaArr[li];
                        float srcR = srcPix ? srcPix[0] : 0.0f;
                        float srcG = srcPix ? srcPix[1] : 0.0f;
                        float srcB = srcPix ? srcPix[2] : 0.0f;
                        float fgLum = luminance(srcR * a, srcG * a, srcB * a);
                        guideArr[li] = fgLum * (1.0f - _edgeProtect) + a * _edgeProtect;
                        if (srcPix) srcPix += 4;
                    }
                }
            }

            // Compute products
            for (int i = 0; i < numPix; i++) {
                meanI[i]  = guideArr[i];
                meanP[i]  = rawAlphaArr[i];
                meanIp[i] = guideArr[i] * rawAlphaArr[i];
                meanII[i] = guideArr[i] * guideArr[i];
            }

            // Blur all four (true Gaussian)
            cpuGaussianBlur(meanI,  scratch, gWeights, w, h, r);
            cpuGaussianBlur(meanP,  scratch, gWeights, w, h, r);
            cpuGaussianBlur(meanIp, scratch, gWeights, w, h, r);
            cpuGaussianBlur(meanII, scratch, gWeights, w, h, r);

            // Compute coefficients a, b
            for (int i = 0; i < numPix; i++) {
                float varI  = meanII[i] - meanI[i] * meanI[i];
                float covIp = meanIp[i] - meanI[i] * meanP[i];
                float a = covIp / (varI + eps);
                float b = meanP[i] - a * meanI[i];
                meanI[i] = a;
                meanP[i] = b;
            }

            // Blur a and b → mean_a, mean_b
            cpuGaussianBlur(meanI, scratch, gWeights, w, h, r);
            cpuGaussianBlur(meanP, scratch, gWeights, w, h, r);

            if (iter < numIter - 1) {
                // Intermediate: evaluate and feed back as new p for next iteration
                for (int i = 0; i < numPix; i++) {
                    rawAlphaArr[i] = fmaxf(0.0f, fminf(1.0f,
                        meanI[i] * guideArr[i] + meanP[i]));
                }
            }
        }

        // Final apply: mix refined matte against saved raw alpha, premultiply
        float mix = _guidedMix;
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
        {
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
            {
                int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                float rawA = savedRawAlpha[li];
                float guidedA = fmaxf(0.0f, fminf(1.0f,
                    meanI[li] * guideArr[li] + meanP[li]));
                float alpha = rawA * (1.0f - mix) + guidedA * mix;

                if (_premultiply) {
                    dstPix[0] *= alpha;
                    dstPix[1] *= alpha;
                    dstPix[2] *= alpha;
                }
                dstPix[3] = alpha;
                dstPix += 4;
            }
        }

        delete[] gWeights;
        delete[] savedRawAlpha;
    }
    else if (_premultiply && !_effect.abort())
    {
        // No GF — just premultiply
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
        {
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
            {
                float a = dstPix[3];
                dstPix[0] *= a;
                dstPix[1] *= a;
                dstPix[2] *= a;
                dstPix += 4;
            }
        }
    }

    // ── PASS 2.5: Edge Color Correction (CPU) ─────────────────────────
    // Uses the final alpha to re-estimate FG color at semi-transparent edges
    // via the matting equation: fg = (src - screen*(1-alpha)) / alpha
    if (_edgeColorCorrect > 0.0f && !_effect.abort())
    {
        float eccAmount = _edgeColorCorrect;
        bool isPremult = (_premultiply != 0);

        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
        {
            if (_effect.abort()) break;
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
            {
                float alpha = dstPix[3];

                if (alpha > 0.005f && alpha < 0.995f)
                {
                    float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);
                    float srcR = srcPix ? srcPix[0] : 0.0f;
                    float srcG = srcPix ? srcPix[1] : 0.0f;
                    float srcB = srcPix ? srcPix[2] : 0.0f;

                    float scrR, scrG, scrB;
                    if (_useScreenInput && _screenImg) {
                        float* scrPix = static_cast<float*>(_screenImg->getPixelAddress(x, y));
                        if (scrPix) { scrR = scrPix[0]; scrG = scrPix[1]; scrB = scrPix[2]; }
                        else { scrR = _pickR; scrG = _pickG; scrB = _pickB; }
                    } else {
                        scrR = _pickR; scrG = _pickG; scrB = _pickB;
                    }

                    // Solve: fg = (src - screen * (1-alpha)) / alpha
                    float invA = 1.0f / alpha;
                    float fgR = (srcR - scrR * (1.0f - alpha)) * invA;
                    float fgG = (srcG - scrG * (1.0f - alpha)) * invA;
                    float fgB = (srcB - scrB * (1.0f - alpha)) * invA;

                    fgR = fmaxf(-0.5f, fminf(2.0f, fgR));
                    fgG = fmaxf(-0.5f, fminf(2.0f, fgG));
                    fgB = fmaxf(-0.5f, fminf(2.0f, fgB));

                    float curR = dstPix[0];
                    float curG = dstPix[1];
                    float curB = dstPix[2];

                    if (isPremult) {
                        curR *= invA; curG *= invA; curB *= invA;
                    }

                    // Edge factor: bell curve peaking at alpha=0.5
                    float ef = alpha * (1.0f - alpha) * 4.0f * eccAmount;

                    float outR = curR + (fgR - curR) * ef;
                    float outG = curG + (fgG - curG) * ef;
                    float outB = curB + (fgB - curB) * ef;

                    if (isPremult) {
                        outR *= alpha; outG *= alpha; outB *= alpha;
                    }

                    dstPix[0] = outR;
                    dstPix[1] = outG;
                    dstPix[2] = outB;
                }

                dstPix += 4;
            }
        }
    }

    // ── PASS 3: Background Wrap (CPU) ─────────────────────────────────
    // Blurs the BG image and bleeds it into the FG edges where alpha < 1.
    // This helps the keyed FG naturally integrate with the new background.
    if (_bgWrapEnabled && _bgImg && _bgWrapAmount > 0.0f && !_effect.abort())
    {
        int bwR = std::max(1, _bgWrapBlur);
        float bwAmount = _bgWrapAmount;

        // Pre-compute Gaussian weights for BG blur
        int bwKernelSize = 2 * bwR + 1;
        float bwSigma = fmaxf(bwR / 3.0f, 0.5f);
        float bwInv2s2 = 1.0f / (2.0f * bwSigma * bwSigma);
        float* bwWeights = new float[bwKernelSize];
        float bwSum = 0.0f;
        for (int i = -bwR; i <= bwR; i++) {
            float wt = expf(-(float)(i * i) * bwInv2s2);
            bwWeights[i + bwR] = wt;
            bwSum += wt;
        }
        for (int i = 0; i < bwKernelSize; i++) bwWeights[i] /= bwSum;

        // Extract BG channels and blur each one
        float* bgR = new float[numPix];
        float* bgG = new float[numPix];
        float* bgB = new float[numPix];
        float* bgScratch = new float[numPix];

        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
            float* bgPix = static_cast<float*>(_bgImg->getPixelAddress(p_ProcWindow.x1, y));
            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
                int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                if (bgPix) {
                    bgR[li] = bgPix[0];
                    bgG[li] = bgPix[1];
                    bgB[li] = bgPix[2];
                    bgPix += 4;
                } else {
                    bgR[li] = 0.0f; bgG[li] = 0.0f; bgB[li] = 0.0f;
                }
            }
        }

        cpuGaussianBlur(bgR, bgScratch, bwWeights, w, h, bwR);
        cpuGaussianBlur(bgG, bgScratch, bwWeights, w, h, bwR);
        cpuGaussianBlur(bgB, bgScratch, bwWeights, w, h, bwR);

        // Apply wrap: blend blurred BG into FG edges weighted by (1-alpha)
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
                int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                float alpha = dstPix[3];
                float wrapWeight = alpha * (1.0f - alpha) * 4.0f * bwAmount;
                dstPix[0] += bgR[li] * wrapWeight;
                dstPix[1] += bgG[li] * wrapWeight;
                dstPix[2] += bgB[li] * wrapWeight;
                dstPix += 4;
            }
        }

        delete[] bgR;
        delete[] bgG;
        delete[] bgB;
        delete[] bgScratch;
        delete[] bwWeights;
    }

    // Cleanup
    delete[] rawAlphaArr;
    delete[] guideArr;
    delete[] meanI;
    delete[] meanP;
    delete[] meanIp;
    delete[] meanII;
    delete[] scratch;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN PLUGIN CLASS
////////////////////////////////////////////////////////////////////////////////

class IBKeymaster : public OFX::ImageEffect
{
public:
    explicit IBKeymaster(OfxImageEffectHandle p_Handle);

    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    void setEnabledness();
    void setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args);

private:
    // Clips
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    OFX::Clip* m_ScreenClip;
    OFX::Clip* m_BgClip;

    // Screen params
    OFX::ChoiceParam*   m_ScreenColor;
    OFX::BooleanParam*  m_UseScreenInput;
    OFX::RGBParam*      m_PickColor;

    // Keyer params
    OFX::DoubleParam*   m_Bias;
    OFX::DoubleParam*   m_Limit;
    OFX::RGBParam*      m_RespillColor;
    OFX::BooleanParam*  m_Premultiply;

    // Matte controls
    OFX::DoubleParam*   m_BlackClip;
    OFX::DoubleParam*   m_WhiteClip;
    OFX::DoubleParam*   m_MatteGamma;

    // Near Grey
    OFX::BooleanParam*  m_NearGreyExtract;
    OFX::DoubleParam*   m_NearGreyAmount;
    OFX::DoubleParam*   m_NearGreySoftness;

    // Guided Filter
    OFX::BooleanParam*  m_GuidedFilterEnabled;
    OFX::IntParam*      m_GuidedRadius;
    OFX::DoubleParam*   m_GuidedEpsilon;
    OFX::DoubleParam*   m_GuidedMix;
    OFX::DoubleParam*   m_EdgeProtect;
    OFX::IntParam*      m_RefineIterations;
    OFX::DoubleParam*   m_EdgeColorCorrect;

    // Background Wrap
    OFX::BooleanParam*  m_BgWrapEnabled;
    OFX::IntParam*      m_BgWrapBlur;
    OFX::DoubleParam*   m_BgWrapAmount;
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////

IBKeymaster::IBKeymaster(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip    = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip    = fetchClip(kOfxImageEffectSimpleSourceClipName);
    m_ScreenClip = fetchClip("Screen");
    m_BgClip     = fetchClip("Background");

    m_ScreenColor     = fetchChoiceParam("screenColor");
    m_UseScreenInput  = fetchBooleanParam("useScreenInput");
    m_PickColor       = fetchRGBParam("pickColor");

    m_Bias            = fetchDoubleParam("bias");
    m_Limit           = fetchDoubleParam("limit");
    m_RespillColor    = fetchRGBParam("respillColor");
    m_Premultiply     = fetchBooleanParam("premultiply");

    m_BlackClip       = fetchDoubleParam("blackClip");
    m_WhiteClip       = fetchDoubleParam("whiteClip");
    m_MatteGamma      = fetchDoubleParam("matteGamma");

    m_NearGreyExtract = fetchBooleanParam("nearGreyExtract");
    m_NearGreyAmount  = fetchDoubleParam("nearGreyAmount");
    m_NearGreySoftness = fetchDoubleParam("nearGreySoftness");

    m_GuidedFilterEnabled = fetchBooleanParam("guidedFilterEnabled");
    m_GuidedRadius    = fetchIntParam("guidedRadius");
    m_GuidedEpsilon   = fetchDoubleParam("guidedEpsilon");
    m_GuidedMix       = fetchDoubleParam("guidedMix");
    m_EdgeProtect     = fetchDoubleParam("edgeProtect");
    m_RefineIterations = fetchIntParam("refineIterations");
    m_EdgeColorCorrect = fetchDoubleParam("edgeColorCorrect");

    m_BgWrapEnabled = fetchBooleanParam("bgWrapEnabled");
    m_BgWrapBlur    = fetchIntParam("bgWrapBlur");
    m_BgWrapAmount  = fetchDoubleParam("bgWrapAmount");

    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER
////////////////////////////////////////////////////////////////////////////////

void IBKeymaster::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) &&
        (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageProcessor processor(*this);
        setupAndProcess(processor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

////////////////////////////////////////////////////////////////////////////////
// IDENTITY CHECK
////////////////////////////////////////////////////////////////////////////////

bool IBKeymaster::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////

void IBKeymaster::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    if (p_ParamName == "useScreenInput" || p_ParamName == "guidedFilterEnabled" ||
        p_ParamName == "nearGreyExtract" || p_ParamName == "bgWrapEnabled")
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////

void IBKeymaster::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT
////////////////////////////////////////////////////////////////////////////////

void IBKeymaster::setEnabledness()
{
    bool useScrIn = m_UseScreenInput->getValue();
    m_PickColor->setEnabled(!useScrIn);

    bool gfOn = m_GuidedFilterEnabled->getValue();
    m_GuidedRadius->setEnabled(gfOn);
    m_GuidedEpsilon->setEnabled(gfOn);
    m_GuidedMix->setEnabled(gfOn);
    m_EdgeProtect->setEnabled(gfOn);
    m_RefineIterations->setEnabled(gfOn);

    bool ngeOn = m_NearGreyExtract->getValue();
    m_NearGreyAmount->setEnabled(ngeOn);
    m_NearGreySoftness->setEnabled(ngeOn);

    bool bgOn = m_BgWrapEnabled->getValue();
    m_BgWrapBlur->setEnabled(bgOn);
    m_BgWrapAmount->setEnabled(bgOn);
}

////////////////////////////////////////////////////////////////////////////////
// SETUP AND PROCESS
////////////////////////////////////////////////////////////////////////////////

void IBKeymaster::setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args)
{
    // Use unique_ptr to auto-release fetched images when this scope exits
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    std::unique_ptr<OFX::Image> screen;
    if (m_ScreenClip && m_ScreenClip->isConnected())
    {
        screen.reset(m_ScreenClip->fetchImage(p_Args.time));
    }

    std::unique_ptr<OFX::Image> background;
    if (m_BgClip && m_BgClip->isConnected())
    {
        background.reset(m_BgClip->fetchImage(p_Args.time));
    }

    // Fetch parameters
    int screenColor;
    m_ScreenColor->getValueAtTime(p_Args.time, screenColor);
    bool useScreenInput = m_UseScreenInput->getValueAtTime(p_Args.time);

    double pickR, pickG, pickB;
    m_PickColor->getValueAtTime(p_Args.time, pickR, pickG, pickB);

    double bias = m_Bias->getValueAtTime(p_Args.time);
    double limit = m_Limit->getValueAtTime(p_Args.time);

    double respillR, respillG, respillB;
    m_RespillColor->getValueAtTime(p_Args.time, respillR, respillG, respillB);

    bool premultiply = m_Premultiply->getValueAtTime(p_Args.time);

    double blackClip = m_BlackClip->getValueAtTime(p_Args.time);
    double whiteClip = m_WhiteClip->getValueAtTime(p_Args.time);
    double matteGamma = m_MatteGamma->getValueAtTime(p_Args.time);

    bool nearGreyExtract = m_NearGreyExtract->getValueAtTime(p_Args.time);
    double nearGreyAmount = m_NearGreyAmount->getValueAtTime(p_Args.time);
    double nearGreySoftness = m_NearGreySoftness->getValueAtTime(p_Args.time);

    bool guidedFilterEnabled = m_GuidedFilterEnabled->getValueAtTime(p_Args.time);
    int guidedRadius = m_GuidedRadius->getValueAtTime(p_Args.time);
    double guidedEpsilon = m_GuidedEpsilon->getValueAtTime(p_Args.time);
    double guidedMix = m_GuidedMix->getValueAtTime(p_Args.time);
    double edgeProtect = m_EdgeProtect->getValueAtTime(p_Args.time);
    int refineIterations = m_RefineIterations->getValueAtTime(p_Args.time);
    double edgeColorCorrect = m_EdgeColorCorrect->getValueAtTime(p_Args.time);

    bool bgWrapEnabled = m_BgWrapEnabled->getValueAtTime(p_Args.time);
    int bgWrapBlur = m_BgWrapBlur->getValueAtTime(p_Args.time);
    double bgWrapAmount = m_BgWrapAmount->getValueAtTime(p_Args.time);

    // Configure processor
    p_Processor.setDstImg(dst.get());
    p_Processor.setSrcImg(src.get());
    p_Processor.setScreenImg(screen.get());
    p_Processor.setBgImg(background.get());
    p_Processor.setGPURenderArgs(p_Args);
    p_Processor.setRenderWindow(p_Args.renderWindow);

    p_Processor.setKeyerParams(
        screenColor, useScreenInput ? 1 : 0,
        (float)pickR, (float)pickG, (float)pickB,
        (float)bias, (float)limit,
        (float)respillR, (float)respillG, (float)respillB,
        premultiply ? 1 : 0, nearGreyExtract ? 1 : 0,
        (float)nearGreyAmount, (float)nearGreySoftness,
        (float)blackClip, (float)whiteClip, (float)matteGamma,
        guidedFilterEnabled ? 1 : 0, guidedRadius,
        (float)guidedEpsilon, (float)guidedMix,
        (float)edgeProtect, refineIterations, (float)edgeColorCorrect,
        bgWrapEnabled ? 1 : 0, bgWrapBlur, (float)bgWrapAmount);

    p_Processor.process();
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY
////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

IBKeymasterFactory::IBKeymasterFactory()
    : OFX::PluginFactoryHelper<IBKeymasterFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

////////////////////////////////////////////////////////////////////////////////
// DESCRIBE
////////////////////////////////////////////////////////////////////////////////

void IBKeymasterFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // GPU support
    p_Desc.setSupportsOpenCLRender(true);
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    p_Desc.setNoSpatialAwareness(true);
}

////////////////////////////////////////////////////////////////////////////////
// DESCRIBE IN CONTEXT — CLIPS + PARAMETERS
////////////////////////////////////////////////////////////////////////////////

static DoubleParamDescriptor* defineDoubleParam(OFX::ImageEffectDescriptor& p_Desc,
                                                const std::string& p_Name,
                                                const std::string& p_Label,
                                                const std::string& p_Hint,
                                                GroupParamDescriptor* p_Parent = nullptr,
                                                double defaultValue = 1.0,
                                                double minValue = 0.0,
                                                double maxValue = 10.0,
                                                double increment = 0.01)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(defaultValue);
    param->setRange(minValue, maxValue);
    param->setIncrement(increment);
    param->setDisplayRange(minValue, maxValue);
    param->setDoubleType(eDoubleTypePlain);
    if (p_Parent) { param->setParent(*p_Parent); }
    return param;
}

void IBKeymasterFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip (foreground)
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Screen clip (optional clean plate)
    ClipDescriptor* screenClip = p_Desc.defineClip("Screen");
    screenClip->addSupportedComponent(ePixelComponentRGBA);
    screenClip->addSupportedComponent(ePixelComponentRGB);
    screenClip->setTemporalClipAccess(false);
    screenClip->setSupportsTiles(kSupportsTiles);
    screenClip->setOptional(true);
    screenClip->setIsMask(false);

    // Background clip (optional — for BG wrap / light wrap)
    ClipDescriptor* bgClip = p_Desc.defineClip("Background");
    bgClip->addSupportedComponent(ePixelComponentRGBA);
    bgClip->addSupportedComponent(ePixelComponentRGB);
    bgClip->setTemporalClipAccess(false);
    bgClip->setSupportsTiles(kSupportsTiles);
    bgClip->setOptional(true);
    bgClip->setIsMask(false);

    // Output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Page
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // ── Group: Screen Settings ──────────────────────────────────────────
    GroupParamDescriptor* screenGroup = p_Desc.defineGroupParam("ScreenGroup");
    screenGroup->setHint("Screen and keying parameters");
    screenGroup->setLabels("Screen Settings", "Screen Settings", "Screen Settings");

    {
        ChoiceParamDescriptor* param = p_Desc.defineChoiceParam("screenColor");
        param->setLabel("Screen Color");
        param->setHint("Dominant chroma of the backing screen.");
        param->appendOption("Red");
        param->appendOption("Green");
        param->appendOption("Blue");
        param->setDefault(kScreenBlue);
        param->setAnimates(true);
        param->setParent(*screenGroup);
        page->addChild(*param);
    }

    {
        BooleanParamDescriptor* param = p_Desc.defineBooleanParam("useScreenInput");
        param->setDefault(true);
        param->setHint("When enabled, reads screen colour from the Screen clip. "
                        "When disabled, uses the Pick Color constant.");
        param->setLabels("Use Screen Input", "Use Screen Input", "Use Screen Input");
        param->setParent(*screenGroup);
        page->addChild(*param);
    }

    {
        RGBParamDescriptor* param = p_Desc.defineRGBParam("pickColor");
        param->setLabels("Pick Color", "Pick Color", "Pick Color");
        param->setHint("Constant screen colour when Screen input is not connected.");
        param->setDefault(0.0, 0.0, 0.0);
        param->setParent(*screenGroup);
        page->addChild(*param);
    }

    // ── Group: Keyer Controls ───────────────────────────────────────────
    GroupParamDescriptor* keyerGroup = p_Desc.defineGroupParam("KeyerGroup");
    keyerGroup->setHint("Keying and despill controls");
    keyerGroup->setLabels("Keyer Controls", "Keyer Controls", "Keyer Controls");

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "bias", "Bias",
            "Weighting between the two complement channels. 0.5 = equal weight.",
            keyerGroup, 0.5, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "limit", "Limit",
            "Scales the despill subtraction. 1.0 = standard.",
            keyerGroup, 1.0, 0.0, 5.0, 0.01);
        page->addChild(*param);
    }

    {
        RGBParamDescriptor* param = p_Desc.defineRGBParam("respillColor");
        param->setLabels("Respill Color", "Respill Color", "Respill Color");
        param->setHint("Colour to add back where screen spill was removed.");
        param->setDefault(0.0, 0.0, 0.0);
        param->setParent(*keyerGroup);
        page->addChild(*param);
    }

    {
        BooleanParamDescriptor* param = p_Desc.defineBooleanParam("premultiply");
        param->setDefault(false);
        param->setHint("Premultiply RGB by alpha for compositing.");
        param->setLabels("Premultiply", "Premultiply", "Premultiply");
        param->setParent(*keyerGroup);
        page->addChild(*param);
    }

    // ── Group: Matte Controls ───────────────────────────────────────────
    GroupParamDescriptor* matteGroup = p_Desc.defineGroupParam("MatteGroup");
    matteGroup->setHint("Matte refinement controls — adjust black/white points of the raw key");
    matteGroup->setLabels("Matte Controls", "Matte Controls", "Matte Controls");

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "blackClip", "Black Clip",
            "Crush blacks in the raw matte. Values below this become fully transparent. "
            "Useful for cleaning up noise in the screen area.",
            matteGroup, 0.0, 0.0, 1.0, 0.001);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "whiteClip", "White Clip",
            "Push whites in the raw matte. Values above this become fully opaque. "
            "Useful for solidifying the foreground core.",
            matteGroup, 1.0, 0.0, 1.0, 0.001);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "matteGamma", "Matte Gamma",
            "Applies a power curve to the alpha after black/white clipping.\n"
            "Values < 1.0 push semi-transparent edges toward opaque (fills in motion blur).\n"
            "Values > 1.0 push them toward transparent (thins edges).\n"
            "1.0 = no change. Particularly useful for recovering natural alpha gradients\n"
            "in motion-blurred and out-of-focus areas.",
            matteGroup, 1.0, 0.1, 4.0, 0.01);
        page->addChild(*param);
    }

    // ── Group: Near Grey Extract ────────────────────────────────────────
    GroupParamDescriptor* ngeGroup = p_Desc.defineGroupParam("NGEGroup");
    ngeGroup->setHint("Near Grey Extraction controls");
    ngeGroup->setLabels("Near Grey Extract", "Near Grey Extract", "Near Grey Extract");

    {
        BooleanParamDescriptor* param = p_Desc.defineBooleanParam("nearGreyExtract");
        param->setDefault(true);
        param->setHint("Improves matte quality in near-grey or ambiguous areas.");
        param->setLabels("Enable", "Enable", "Enable");
        param->setParent(*ngeGroup);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "nearGreyAmount", "Strength",
            "How much the near-grey extraction contributes to the final alpha.\n"
            "At 0.0: no effect (same as disabled) — pure colour-difference matte.\n"
            "At 1.0: full screen-composite of the NGE alpha with the raw alpha.\n"
            "0.3–0.7 is a good range for subtle matte improvement without over-filling.",
            ngeGroup, 0.5, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "nearGreySoftness", "Softness",
            "Controls how the 'greyness' of each pixel is measured internally.\n"
            "At 0.0: uses the dominant screen channel of the despill ratio — "
            "stricter, reacts more to the screen channel residual.\n"
            "At 1.0: uses the brighter complement channel — more forgiving, "
            "better at catching near-grey/ambiguous areas.\n"
            "0.5–1.0 is typical. Lower values for aggressive keying, higher for gentle.",
            ngeGroup, 1.0, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    // ── Group: Guided Filter ────────────────────────────────────────────
    GroupParamDescriptor* gfGroup = p_Desc.defineGroupParam("GuidedFilterGroup");
    gfGroup->setHint("Edge-aware matte refinement using the source luminance as guide");
    gfGroup->setLabels("Guided Filter", "Guided Filter", "Guided Filter");

    {
        BooleanParamDescriptor* param = p_Desc.defineBooleanParam("guidedFilterEnabled");
        param->setDefault(true);
        param->setHint("Enable guided filter matte refinement. Uses source luminance "
                        "as an edge guide to recover hair detail and soft edges.");
        param->setLabels("Enable", "Enable", "Enable");
        param->setParent(*gfGroup);
        page->addChild(*param);
    }

    {
        IntParamDescriptor* param = p_Desc.defineIntParam("guidedRadius");
        param->setLabels("Radius", "Radius", "Radius");
        param->setScriptName("guidedRadius");
        param->setHint("Filter window radius in pixels. Larger values smooth more "
                        "but may lose very fine detail. 4-16 typical for 1080p, "
                        "8-32 for 4K.");
        param->setDefault(8);
        param->setRange(1, 100);
        param->setDisplayRange(1, 50);
        param->setParent(*gfGroup);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "guidedEpsilon", "Epsilon",
            "Edge sensitivity. Smaller values preserve more edges but may "
            "introduce noise. Larger values produce smoother results.\n"
            "Try 0.001 for fine hair, 0.01 for general use, 0.1 for heavy smoothing.",
            gfGroup, 0.01, 0.0001, 1.0, 0.001);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "guidedMix", "Mix",
            "Blend between raw matte (0.0) and guided-filter-refined matte (1.0). "
            "Useful for dialling in the right amount of refinement.",
            gfGroup, 1.0, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "edgeProtect", "Edge Protection",
            "Blends the guide signal from source luminance toward the raw alpha.\n"
            "At 0.0, the guide is pure luminance — best for recovering fine hair detail "
            "but may leave a shadow on solid edges like forearms.\n"
            "At 1.0, the guide is the raw alpha itself (self-guided) — preserves all "
            "hard edges cleanly but relies less on luminance structure for hair.\n"
            "0.5 is a good starting point for mixed foreground subjects.",
            gfGroup, 0.5, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    {
        IntParamDescriptor* param = p_Desc.defineIntParam("refineIterations");
        param->setLabels("Refine Iterations", "Refine Iterations", "Refine Iterations");
        param->setScriptName("refineIterations");
        param->setHint("Number of iterative guided-filter refinement passes.\n"
                        "1 = standard single pass (fast, good for most shots).\n"
                        "2-3 = each pass uses the refined matte to compute a better "
                        "foreground estimate, then feeds that back as the guide signal. "
                        "This progressively cleans up the matte — similar to how a "
                        "neural net keyer iteratively improves its output.\n"
                        "Higher values give diminishing returns and cost more GPU time.");
        param->setDefault(2);
        param->setRange(1, 5);
        param->setDisplayRange(1, 5);
        param->setParent(*gfGroup);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "edgeColorCorrect", "Edge Color Correct",
            "Re-estimates the true foreground colour at semi-transparent edges "
            "using the matting equation: fg = (src - screen*(1-alpha)) / alpha.\n\n"
            "This corrects residual screen contamination that the initial despill "
            "couldn't fully remove, especially at fine hair detail and motion-blurred edges "
            "where the alpha has been refined by the guided filter.\n\n"
            "Higher values apply stronger correction. The effect is concentrated on edge "
            "pixels (bell curve peaking at alpha = 0.5) and leaves fully opaque/transparent "
            "pixels untouched.",
            gfGroup, 0.0, 0.0, 1.0, 0.01);
        page->addChild(*param);
    }

    // ── Group: Background Wrap ───────────────────────────────────────────
    GroupParamDescriptor* bgGroup = p_Desc.defineGroupParam("BgWrapGroup");
    bgGroup->setHint("Bleeds a blurred version of the new background into the foreground edges.\n"
                     "Connect the replacement background to the Background input, then adjust "
                     "blur and amount to taste. This simulates the colour spill the foreground "
                     "would naturally pick up from its new environment, making the comp more "
                     "convincing without a separate light-wrap node.");
    bgGroup->setLabels("Background Wrap", "Background Wrap", "Background Wrap");

    {
        BooleanParamDescriptor* param = p_Desc.defineBooleanParam("bgWrapEnabled");
        param->setDefault(false);
        param->setHint("Enable background wrap. Requires the Background clip to be connected.");
        param->setLabels("Enable", "Enable", "Enable");
        param->setParent(*bgGroup);
        page->addChild(*param);
    }

    {
        IntParamDescriptor* param = p_Desc.defineIntParam("bgWrapBlur");
        param->setLabels("Blur Radius", "Blur Radius", "Blur Radius");
        param->setScriptName("bgWrapBlur");
        param->setHint("Gaussian blur radius applied to the background before wrapping.\n"
                        "Larger values create a softer, wider colour bleed around the edges.\n"
                        "10–30 for subtle wrap, 30–100 for heavy light-wrap looks.");
        param->setDefault(20);
        param->setRange(1, 200);
        param->setDisplayRange(1, 100);
        param->setParent(*bgGroup);
        page->addChild(*param);
    }

    {
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "bgWrapAmount", "Amount",
            "How much blurred background to bleed into the foreground edges.\n"
            "The wrap uses an edge-only weighting (alpha * (1-alpha) * 4) so it peaks at \n"
            "semi-transparent edge pixels and is zero at both fully opaque and fully \n"
            "transparent areas — only the matte transition zone is affected.\n"
            "0.3–0.8 is usually enough for a natural look.",
            bgGroup, 0.5, 0.0, 2.0, 0.01);
        page->addChild(*param);
    }
}

////////////////////////////////////////////////////////////////////////////////
// CREATE INSTANCE
////////////////////////////////////////////////////////////////////////////////

ImageEffect* IBKeymasterFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new IBKeymaster(p_Handle);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION
////////////////////////////////////////////////////////////////////////////////

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static IBKeymasterFactory ibKeymaster;
    p_FactoryArray.push_back(&ibKeymaster);
}
