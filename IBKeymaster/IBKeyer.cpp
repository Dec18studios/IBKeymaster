/////// IBKeyer v2 OFX Plugin
// Guided-Filter Enhanced Image-Based Keyer
// Port of Jed Smith's IBKeyer + guided filter matte refinement + matte controls
#include "IBKeyer.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
#define kPluginName "IBKeyer"
#define kPluginGrouping "create@Dec18Studios.com"
#define kPluginDescription \
    "Image-Based Keyer with Guided Filter refinement.\n\n" \
    "Extracts a high-quality matte and despilled foreground from green/blue screen footage " \
    "by comparing source pixels against a clean screen plate or pick colour.\n\n" \
    "The Guided Filter uses the source luminance as an edge-aware guide to refine " \
    "the raw colour-difference matte, recovering hair detail, transparency, " \
    "and motion blur that traditional per-pixel keyers lose.\n\n" \
    "Based on IBKeyer by Jed Smith (gaffer-tools) + He et al. guided filter."
#define kPluginIdentifier "com.OpenFXSample.IBKeyer"
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
    void setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4);

    void setKeyerParams(int p_ScreenColor, int p_UseScreenInput,
                        float p_PickR, float p_PickG, float p_PickB,
                        float p_Bias, float p_Limit,
                        float p_RespillR, float p_RespillG, float p_RespillB,
                        int p_Premultiply, int p_NearGreyExtract, float p_NearGreyAmount,
                        float p_BlackClip, float p_WhiteClip,
                        int p_GuidedFilterEnabled, int p_GuidedRadius,
                        float p_GuidedEpsilon, float p_GuidedMix);

private:
    OFX::Image* _srcImg;
    OFX::Image* _screenImg;
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

    // Matte controls
    float _blackClip;
    float _whiteClip;

    // Guided filter params
    int   _guidedFilterEnabled;
    int   _guidedRadius;
    float _guidedEpsilon;
    float _guidedMix;

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
        float s0 = _nearGreyAmount;
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
                                    int p_Premultiply, int p_NearGreyExtract, float p_NearGreyAmount,
                                    float p_BlackClip, float p_WhiteClip,
                                    int p_GuidedFilterEnabled, int p_GuidedRadius,
                                    float p_GuidedEpsilon, float p_GuidedMix)
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
    _blackClip       = p_BlackClip;
    _whiteClip       = p_WhiteClip;
    _guidedFilterEnabled = p_GuidedFilterEnabled;
    _guidedRadius    = p_GuidedRadius;
    _guidedEpsilon   = p_GuidedEpsilon;
    _guidedMix       = p_GuidedMix;
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
                          int p_Premultiply, int p_NearGreyExtract, float p_NearGreyAmount,
                          float p_BlackClip, float p_WhiteClip,
                          int p_GuidedFilterEnabled, int p_GuidedRadius,
                          float p_GuidedEpsilon, float p_GuidedMix,
                          const float* p_Input, const float* p_Screen, float* p_Output);
#endif

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                           int p_ScreenColor, int p_UseScreenInput,
                           float p_PickR, float p_PickG, float p_PickB,
                           float p_Bias, float p_Limit,
                           float p_RespillR, float p_RespillG, float p_RespillB,
                           int p_Premultiply, int p_NearGreyExtract, float p_NearGreyAmount,
                           float p_BlackClip, float p_WhiteClip,
                           int p_GuidedFilterEnabled, int p_GuidedRadius,
                           float p_GuidedEpsilon, float p_GuidedMix,
                           const float* p_Input, const float* p_Screen, float* p_Output);
#endif

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                            int p_ScreenColor, int p_UseScreenInput,
                            float p_PickR, float p_PickG, float p_PickB,
                            float p_Bias, float p_Limit,
                            float p_RespillR, float p_RespillG, float p_RespillB,
                            int p_Premultiply, int p_NearGreyExtract, float p_NearGreyAmount,
                            float p_BlackClip, float p_WhiteClip,
                            int p_GuidedFilterEnabled, int p_GuidedRadius,
                            float p_GuidedEpsilon, float p_GuidedMix,
                            const float* p_Input, const float* p_Screen, float* p_Output);

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

    RunCudaKernel(_pCudaStream, width, height,
                  _screenColor, _useScreenInput,
                  _pickR, _pickG, _pickB,
                  _bias, _limit,
                  _respillR, _respillG, _respillB,
                  _premultiply, _nearGreyExtract, _nearGreyAmount,
                  _blackClip, _whiteClip,
                  _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                  input, screen, output);
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

    RunMetalKernel(_pMetalCmdQ, width, height,
                   _screenColor, _useScreenInput,
                   _pickR, _pickG, _pickB,
                   _bias, _limit,
                   _respillR, _respillG, _respillB,
                   _premultiply, _nearGreyExtract, _nearGreyAmount,
                   _blackClip, _whiteClip,
                   _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                   input, screen, output);
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

    RunOpenCLKernel(_pOpenCLCmdQ, width, height,
                    _screenColor, _useScreenInput,
                    _pickR, _pickG, _pickB,
                    _bias, _limit,
                    _respillR, _respillG, _respillB,
                    _premultiply, _nearGreyExtract, _nearGreyAmount,
                    _blackClip, _whiteClip,
                    _guidedFilterEnabled, _guidedRadius, _guidedEpsilon, _guidedMix,
                    input, screen, output);
}

////////////////////////////////////////////////////////////////////////////////
// CPU PROCESSING — FALLBACK (includes guided filter on CPU)
////////////////////////////////////////////////////////////////////////////////

// CPU box blur helper (single channel, in-place via scratch buffer)
static void cpuBoxBlur(float* data, float* scratch, int w, int h, int radius)
{
    // 3 iterations of separable box blur → approximate Gaussian
    for (int iter = 0; iter < 3; iter++) {
        // Horizontal pass: data → scratch
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                int count = 0;
                for (int dx = -radius; dx <= radius; dx++) {
                    int sx = std::max(0, std::min(w - 1, x + dx));
                    sum += data[y * w + sx];
                    count++;
                }
                scratch[y * w + x] = sum / (float)count;
            }
        }
        // Vertical pass: scratch → data
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                int count = 0;
                for (int dy = -radius; dy <= radius; dy++) {
                    int sy = std::max(0, std::min(h - 1, y + dy));
                    sum += scratch[sy * w + x];
                    count++;
                }
                data[y * w + x] = sum / (float)count;
            }
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

                if (_nearGreyExtract)
                {
                    float divR = safeDivide(ssR, srcR);
                    float divG = safeDivide(ssG, srcG);
                    float divB = safeDivide(ssB, srcB);
                    float ngeAlpha = despillNGE(divR, divG, divB);
                    alpha = ngeAlpha + alpha - ngeAlpha * alpha;
                }

                // Black/White clip
                float lo = _blackClip;
                float hi = _whiteClip;
                if (hi > lo + 1e-6f) {
                    alpha = fmaxf(0.0f, fminf(1.0f, (alpha - lo) / (hi - lo)));
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
                    guideArr[li] = luminance(srcR, srcG, srcB);
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

    // ── PASS 2: Guided filter (CPU) ─────────────────────────────────────
    if (doGF && !_effect.abort())
    {
        int r = _guidedRadius;
        float eps = _guidedEpsilon;

        // Compute products
        for (int i = 0; i < numPix; i++) {
            meanI[i]  = guideArr[i];
            meanP[i]  = rawAlphaArr[i];
            meanIp[i] = guideArr[i] * rawAlphaArr[i];
            meanII[i] = guideArr[i] * guideArr[i];
        }

        // Blur all four
        cpuBoxBlur(meanI,  scratch, w, h, r);
        cpuBoxBlur(meanP,  scratch, w, h, r);
        cpuBoxBlur(meanIp, scratch, w, h, r);
        cpuBoxBlur(meanII, scratch, w, h, r);

        // Compute coefficients a, b
        for (int i = 0; i < numPix; i++) {
            float varI  = meanII[i] - meanI[i] * meanI[i];
            float covIp = meanIp[i] - meanI[i] * meanP[i];
            float a = covIp / (varI + eps);
            float b = meanP[i] - a * meanI[i];
            meanI[i] = a;   // reuse buffer
            meanP[i] = b;   // reuse buffer
        }

        // Blur a and b → mean_a, mean_b
        cpuBoxBlur(meanI, scratch, w, h, r);  // mean_a
        cpuBoxBlur(meanP, scratch, w, h, r);  // mean_b

        // Apply refined matte
        float mix = _guidedMix;
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
        {
            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
            {
                int li = (y - p_ProcWindow.y1) * w + (x - p_ProcWindow.x1);
                float rawA = dstPix[3];
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

class IBKeyer : public OFX::ImageEffect
{
public:
    explicit IBKeyer(OfxImageEffectHandle p_Handle);

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

    // Near Grey
    OFX::BooleanParam*  m_NearGreyExtract;
    OFX::DoubleParam*   m_NearGreyAmount;

    // Guided Filter
    OFX::BooleanParam*  m_GuidedFilterEnabled;
    OFX::IntParam*      m_GuidedRadius;
    OFX::DoubleParam*   m_GuidedEpsilon;
    OFX::DoubleParam*   m_GuidedMix;
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////

IBKeyer::IBKeyer(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip    = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip    = fetchClip(kOfxImageEffectSimpleSourceClipName);
    m_ScreenClip = fetchClip("Screen");

    m_ScreenColor     = fetchChoiceParam("screenColor");
    m_UseScreenInput  = fetchBooleanParam("useScreenInput");
    m_PickColor       = fetchRGBParam("pickColor");

    m_Bias            = fetchDoubleParam("bias");
    m_Limit           = fetchDoubleParam("limit");
    m_RespillColor    = fetchRGBParam("respillColor");
    m_Premultiply     = fetchBooleanParam("premultiply");

    m_BlackClip       = fetchDoubleParam("blackClip");
    m_WhiteClip       = fetchDoubleParam("whiteClip");

    m_NearGreyExtract = fetchBooleanParam("nearGreyExtract");
    m_NearGreyAmount  = fetchDoubleParam("nearGreyAmount");

    m_GuidedFilterEnabled = fetchBooleanParam("guidedFilterEnabled");
    m_GuidedRadius    = fetchIntParam("guidedRadius");
    m_GuidedEpsilon   = fetchDoubleParam("guidedEpsilon");
    m_GuidedMix       = fetchDoubleParam("guidedMix");

    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER
////////////////////////////////////////////////////////////////////////////////

void IBKeyer::render(const OFX::RenderArguments& p_Args)
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

bool IBKeyer::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////

void IBKeyer::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    if (p_ParamName == "useScreenInput" || p_ParamName == "guidedFilterEnabled" ||
        p_ParamName == "nearGreyExtract")
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////

void IBKeyer::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT
////////////////////////////////////////////////////////////////////////////////

void IBKeyer::setEnabledness()
{
    bool useScrIn = m_UseScreenInput->getValue();
    m_PickColor->setEnabled(!useScrIn);

    bool gfOn = m_GuidedFilterEnabled->getValue();
    m_GuidedRadius->setEnabled(gfOn);
    m_GuidedEpsilon->setEnabled(gfOn);
    m_GuidedMix->setEnabled(gfOn);

    bool ngeOn = m_NearGreyExtract->getValue();
    m_NearGreyAmount->setEnabled(ngeOn);
}

////////////////////////////////////////////////////////////////////////////////
// SETUP AND PROCESS
////////////////////////////////////////////////////////////////////////////////

void IBKeyer::setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args)
{
    OFX::Image* dst = m_DstClip->fetchImage(p_Args.time);
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    OFX::Image* src = m_SrcClip->fetchImage(p_Args.time);
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    OFX::Image* screen = nullptr;
    if (m_ScreenClip && m_ScreenClip->isConnected())
    {
        screen = m_ScreenClip->fetchImage(p_Args.time);
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

    bool nearGreyExtract = m_NearGreyExtract->getValueAtTime(p_Args.time);
    double nearGreyAmount = m_NearGreyAmount->getValueAtTime(p_Args.time);

    bool guidedFilterEnabled = m_GuidedFilterEnabled->getValueAtTime(p_Args.time);
    int guidedRadius = m_GuidedRadius->getValueAtTime(p_Args.time);
    double guidedEpsilon = m_GuidedEpsilon->getValueAtTime(p_Args.time);
    double guidedMix = m_GuidedMix->getValueAtTime(p_Args.time);

    // Configure processor
    p_Processor.setDstImg(dst);
    p_Processor.setSrcImg(src);
    p_Processor.setScreenImg(screen);
    p_Processor.setGPURenderArgs(p_Args);
    p_Processor.setRenderWindow(p_Args.renderWindow);

    p_Processor.setKeyerParams(
        screenColor, useScreenInput ? 1 : 0,
        (float)pickR, (float)pickG, (float)pickB,
        (float)bias, (float)limit,
        (float)respillR, (float)respillG, (float)respillB,
        premultiply ? 1 : 0, nearGreyExtract ? 1 : 0, (float)nearGreyAmount,
        (float)blackClip, (float)whiteClip,
        guidedFilterEnabled ? 1 : 0, guidedRadius,
        (float)guidedEpsilon, (float)guidedMix);

    p_Processor.process();
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY
////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

IBKeyerFactory::IBKeyerFactory()
    : OFX::PluginFactoryHelper<IBKeyerFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

////////////////////////////////////////////////////////////////////////////////
// DESCRIBE
////////////////////////////////////////////////////////////////////////////////

void IBKeyerFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

void IBKeyerFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
        DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "nearGreyAmount", "Amount",
            "Strength of the near-grey extraction.",
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
}

////////////////////////////////////////////////////////////////////////////////
// CREATE INSTANCE
////////////////////////////////////////////////////////////////////////////////

ImageEffect* IBKeyerFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new IBKeyer(p_Handle);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION
////////////////////////////////////////////////////////////////////////////////

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static IBKeyerFactory ibKeyer;
    p_FactoryArray.push_back(&ibKeyer);
}
