/////// IBKeyer v2 OFX Plugin
// Guided-Filter Enhanced Image-Based Keyer
// Port of Jed Smith's IBKeyer + guided filter matte refinement + matte controls
//
// note:
// This file used to be a monolithic plugin + CPU/GPU implementation. During the cross-platform
// port we split it so the host glue stays here, while the moved processing sections now live in:
//   - IBKeyerBackend.cpp : render request translation, CPU fallback, backend routing
//   - IBKeyerCuda.cu     : CUDA kernels and host-CUDA zero-copy / staged CUDA execution
// The section headers below intentionally echo the original file so it is easier to follow what
// moved where and why.
#include "IBKeyer.h"

#include <memory>
#include <string>

#include "IBKeyerBackend.h"
#include "ofxsImageEffect.h"
#include "ofxsLog.h"

namespace {

////////////////////////////////////////////////////////////////////////////////
// PLUGIN DESCRIPTION + CONSTANTS
////////////////////////////////////////////////////////////////////////////////

constexpr const char* kPluginName = "IBKeyer";
constexpr const char* kPluginGrouping = "create@Dec18Studios.com";
constexpr const char* kPluginDescription =
    "Image-Based Keyer with Guided Filter refinement.\n\n"
    "Extracts a high-quality matte and despilled foreground from green/blue screen footage "
    "by comparing source pixels against a clean screen plate or pick colour.\n\n"
    "The Guided Filter uses the source luminance as an edge-aware guide to refine "
    "the raw colour-difference matte, recovering hair detail, transparency, "
    "and motion blur that traditional per-pixel keyers lose.\n\n"
    "Based on IBKeyer by Jed Smith (gaffer-tools) + He et al. guided filter.";
constexpr const char* kPluginIdentifier = "com.OpenFXSample.IBKeyer";
constexpr int kPluginVersionMajor = 2;
// This changed the public OFX surface in a non-trivial way: new clips, new params, and new
// backend-routing rules. Keeping the old version number after that can leave hosts holding onto a
// stale descriptor cache and trying to reconcile it with a different binary. Bumping the version is
// the polite way to tell Resolve "this is materially a new plugin shape, please rescan it fresh."
constexpr int kPluginVersionMinor = 2;
constexpr bool kSupportsTiles = false;
constexpr bool kSupportsMultiResolution = false;
constexpr bool kSupportsMultipleClipPARs = false;

////////////////////////////////////////////////////////////////////////////////
// MAIN PLUGIN CLASS
////////////////////////////////////////////////////////////////////////////////

class IBKeyerPlugin : public OFX::ImageEffect
{
public:
    explicit IBKeyerPlugin(OfxImageEffectHandle p_Handle);

    void render(const OFX::RenderArguments& p_Args) override;
    bool isIdentity(const OFX::IsIdentityArguments& p_Args,
                    OFX::Clip*& p_IdentityClip,
                    double& p_IdentityTime) override;
    void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName) override;
    void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName) override;

private:
    void setEnabledness();

    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    OFX::Clip* m_ScreenClip;
    OFX::Clip* m_BgClip;
    OFX::Clip* m_GarbageMatteClip;
    OFX::Clip* m_OcclusionMatteClip;

    OFX::ChoiceParam* m_ScreenColor;
    OFX::BooleanParam* m_UseScreenInput;
    OFX::RGBParam* m_PickColor;
    OFX::DoubleParam* m_Bias;
    OFX::DoubleParam* m_Limit;
    OFX::RGBParam* m_RespillColor;
    OFX::BooleanParam* m_Premultiply;
    OFX::DoubleParam* m_BlackClip;
    OFX::DoubleParam* m_WhiteClip;
    OFX::DoubleParam* m_MatteGamma;
    OFX::BooleanParam* m_PrematteEnabled;
    OFX::IntParam* m_PrematteBlur;
    OFX::IntParam* m_PrematteErode;
    OFX::IntParam* m_PrematteIterations;
    OFX::BooleanParam* m_NearGreyExtract;
    OFX::DoubleParam* m_NearGreyAmount;
    OFX::DoubleParam* m_NearGreySoftness;
    OFX::BooleanParam* m_GuidedFilterEnabled;
    OFX::ChoiceParam* m_GuidedFilterMode;
    OFX::IntParam* m_GuidedRadius;
    OFX::DoubleParam* m_GuidedEpsilon;
    OFX::DoubleParam* m_GuidedMix;
    OFX::DoubleParam* m_EdgeProtect;
    OFX::IntParam* m_RefineIterations;
    OFX::DoubleParam* m_EdgeColorCorrect;
    OFX::BooleanParam* m_BgWrapEnabled;
    OFX::IntParam* m_BgWrapBlur;
    OFX::DoubleParam* m_BgWrapAmount;
    OFX::BooleanParam* m_AdditiveKeyEnabled;
    OFX::ChoiceParam* m_AdditiveKeyMode;
    OFX::DoubleParam* m_AdditiveKeySaturation;
    OFX::DoubleParam* m_AdditiveKeyAmount;
    OFX::BooleanParam* m_AdditiveKeyBlackClamp;
    OFX::ChoiceParam* m_ViewMode;
};

IBKeyerPlugin::IBKeyerPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
    , m_DstClip(fetchClip(kOfxImageEffectOutputClipName))
    , m_SrcClip(fetchClip(kOfxImageEffectSimpleSourceClipName))
    , m_ScreenClip(fetchClip("Screen"))
    , m_BgClip(fetchClip("Background"))
    , m_GarbageMatteClip(fetchClip("GarbageMatte"))
    , m_OcclusionMatteClip(fetchClip("OcclusionMatte"))
    , m_ScreenColor(fetchChoiceParam("screenColor"))
    , m_UseScreenInput(fetchBooleanParam("useScreenInput"))
    , m_PickColor(fetchRGBParam("pickColor"))
    , m_Bias(fetchDoubleParam("bias"))
    , m_Limit(fetchDoubleParam("limit"))
    , m_RespillColor(fetchRGBParam("respillColor"))
    , m_Premultiply(fetchBooleanParam("premultiply"))
    , m_BlackClip(fetchDoubleParam("blackClip"))
    , m_WhiteClip(fetchDoubleParam("whiteClip"))
    , m_MatteGamma(fetchDoubleParam("matteGamma"))
    , m_PrematteEnabled(fetchBooleanParam("prematteEnabled"))
    , m_PrematteBlur(fetchIntParam("prematteBlur"))
    , m_PrematteErode(fetchIntParam("prematteErode"))
    , m_PrematteIterations(fetchIntParam("prematteIterations"))
    , m_NearGreyExtract(fetchBooleanParam("nearGreyExtract"))
    , m_NearGreyAmount(fetchDoubleParam("nearGreyAmount"))
    , m_NearGreySoftness(fetchDoubleParam("nearGreySoftness"))
    , m_GuidedFilterEnabled(fetchBooleanParam("guidedFilterEnabled"))
    , m_GuidedFilterMode(fetchChoiceParam("guidedFilterMode"))
    , m_GuidedRadius(fetchIntParam("guidedRadius"))
    , m_GuidedEpsilon(fetchDoubleParam("guidedEpsilon"))
    , m_GuidedMix(fetchDoubleParam("guidedMix"))
    , m_EdgeProtect(fetchDoubleParam("edgeProtect"))
    , m_RefineIterations(fetchIntParam("refineIterations"))
    , m_EdgeColorCorrect(fetchDoubleParam("edgeColorCorrect"))
    , m_BgWrapEnabled(fetchBooleanParam("bgWrapEnabled"))
    , m_BgWrapBlur(fetchIntParam("bgWrapBlur"))
    , m_BgWrapAmount(fetchDoubleParam("bgWrapAmount"))
    , m_AdditiveKeyEnabled(fetchBooleanParam("additiveKeyEnabled"))
    , m_AdditiveKeyMode(fetchChoiceParam("additiveKeyMode"))
    , m_AdditiveKeySaturation(fetchDoubleParam("additiveKeySaturation"))
    , m_AdditiveKeyAmount(fetchDoubleParam("additiveKeyAmount"))
    , m_AdditiveKeyBlackClamp(fetchBooleanParam("additiveKeyBlackClamp"))
    , m_ViewMode(fetchChoiceParam("viewMode"))
{
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER
////////////////////////////////////////////////////////////////////////////////
// Moved from: the old "RENDER" + "SETUP AND PROCESS" sections.
//
//  
// The old file fetched images, read params, chose a backend, and processed pixels in one place.
// That made it fast to write initially, but hard to reason about later because host concerns and
// algorithm concerns were tangled together. Splitting it lets us keep this function focused on
// OFX lifecycle work while the backend layer handles CPU/CUDA/Metal details.
//
//   
// this way it removes hidden backend decisions from the host glue, which makes fallback behavior explicit
// and makes it much easier to validate whether a render used host CUDA, staged CUDA, Metal, or CPU.
void IBKeyerPlugin::render(const OFX::RenderArguments& p_Args)
{
    // Get output image.
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));

    // Get source image.
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    if (!dst || !src) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    if (dst->getPixelDepth() != OFX::eBitDepthFloat ||
        src->getPixelDepth() != OFX::eBitDepthFloat ||
        dst->getPixelComponents() != OFX::ePixelComponentRGBA ||
        src->getPixelComponents() != OFX::ePixelComponentRGBA) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    // Get screen image (optional clean plate).
    std::unique_ptr<OFX::Image> screen;
    if (m_ScreenClip && m_ScreenClip->isConnected()) {
        screen.reset(m_ScreenClip->fetchImage(p_Args.time));
        if (screen &&
            (screen->getPixelDepth() != OFX::eBitDepthFloat ||
             (screen->getPixelComponents() != OFX::ePixelComponentRGB &&
              screen->getPixelComponents() != OFX::ePixelComponentRGBA))) {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    // Get background image (optional light-wrap source).
    std::unique_ptr<OFX::Image> background;
    if (m_BgClip && m_BgClip->isConnected()) {
        background.reset(m_BgClip->fetchImage(p_Args.time));
        if (background &&
            (background->getPixelDepth() != OFX::eBitDepthFloat ||
             (background->getPixelComponents() != OFX::ePixelComponentRGB &&
              background->getPixelComponents() != OFX::ePixelComponentRGBA))) {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    // External matte clips are deliberately optional. They are constraints on the matte, not
    // mandatory inputs to the key itself, so disconnecting them should never change routing or
    // make the effect invalid.
    std::unique_ptr<OFX::Image> garbageMatte;
    if (m_GarbageMatteClip && m_GarbageMatteClip->isConnected()) {
        garbageMatte.reset(m_GarbageMatteClip->fetchImage(p_Args.time));
        if (garbageMatte &&
            (garbageMatte->getPixelDepth() != OFX::eBitDepthFloat ||
             (garbageMatte->getPixelComponents() != OFX::ePixelComponentRGB &&
              garbageMatte->getPixelComponents() != OFX::ePixelComponentRGBA))) {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    std::unique_ptr<OFX::Image> occlusionMatte;
    if (m_OcclusionMatteClip && m_OcclusionMatteClip->isConnected()) {
        occlusionMatte.reset(m_OcclusionMatteClip->fetchImage(p_Args.time));
        if (occlusionMatte &&
            (occlusionMatte->getPixelDepth() != OFX::eBitDepthFloat ||
             (occlusionMatte->getPixelComponents() != OFX::ePixelComponentRGB &&
              occlusionMatte->getPixelComponents() != OFX::ePixelComponentRGBA))) {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    // Fetch parameter values once and freeze them into a backend-agnostic request. The old file
    // pushed these directly into an ImageProcessor instance; the split version keeps that same
    // intent, but makes the backend choice explicit and testable.
    int screenColor = 0;
    m_ScreenColor->getValueAtTime(p_Args.time, screenColor);

    double pickR = 0.0;
    double pickG = 0.0;
    double pickB = 0.0;
    m_PickColor->getValueAtTime(p_Args.time, pickR, pickG, pickB);

    double respillR = 0.0;
    double respillG = 0.0;
    double respillB = 0.0;
    m_RespillColor->getValueAtTime(p_Args.time, respillR, respillG, respillB);

    IBKeyerCore::IBKeyerParams params;
    params.screenColor = screenColor;
    params.useScreenInput = m_UseScreenInput->getValueAtTime(p_Args.time) && static_cast<bool>(screen);
    params.pickR = static_cast<float>(pickR);
    params.pickG = static_cast<float>(pickG);
    params.pickB = static_cast<float>(pickB);
    params.bias = static_cast<float>(m_Bias->getValueAtTime(p_Args.time));
    params.limit = static_cast<float>(m_Limit->getValueAtTime(p_Args.time));
    params.respillR = static_cast<float>(respillR);
    params.respillG = static_cast<float>(respillG);
    params.respillB = static_cast<float>(respillB);
    params.premultiply = m_Premultiply->getValueAtTime(p_Args.time);
    params.blackClip = static_cast<float>(m_BlackClip->getValueAtTime(p_Args.time));
    params.whiteClip = static_cast<float>(m_WhiteClip->getValueAtTime(p_Args.time));
    params.matteGamma = static_cast<float>(m_MatteGamma->getValueAtTime(p_Args.time));
    params.prematteEnabled = m_PrematteEnabled->getValueAtTime(p_Args.time);
    params.prematteBlur = m_PrematteBlur->getValueAtTime(p_Args.time);
    params.prematteErode = m_PrematteErode->getValueAtTime(p_Args.time);
    params.prematteIterations = m_PrematteIterations->getValueAtTime(p_Args.time);
    params.nearGreyExtract = m_NearGreyExtract->getValueAtTime(p_Args.time);
    params.nearGreyAmount = static_cast<float>(m_NearGreyAmount->getValueAtTime(p_Args.time));
    params.nearGreySoftness = static_cast<float>(m_NearGreySoftness->getValueAtTime(p_Args.time));
    params.guidedFilterEnabled = m_GuidedFilterEnabled->getValueAtTime(p_Args.time);
    int guidedFilterMode = 0;
    m_GuidedFilterMode->getValueAtTime(p_Args.time, guidedFilterMode);
    params.guidedFilterMode = guidedFilterMode;
    params.guidedRadius = m_GuidedRadius->getValueAtTime(p_Args.time);
    params.guidedEpsilon = static_cast<float>(m_GuidedEpsilon->getValueAtTime(p_Args.time));
    params.guidedMix = static_cast<float>(m_GuidedMix->getValueAtTime(p_Args.time));
    params.edgeProtect = static_cast<float>(m_EdgeProtect->getValueAtTime(p_Args.time));
    params.refineIterations = m_RefineIterations->getValueAtTime(p_Args.time);
    params.edgeColorCorrect = static_cast<float>(m_EdgeColorCorrect->getValueAtTime(p_Args.time));
    params.bgWrapEnabled = m_BgWrapEnabled->getValueAtTime(p_Args.time) && static_cast<bool>(background);
    params.bgWrapBlur = m_BgWrapBlur->getValueAtTime(p_Args.time);
    params.bgWrapAmount = static_cast<float>(m_BgWrapAmount->getValueAtTime(p_Args.time));
    params.additiveKeyEnabled = m_AdditiveKeyEnabled->getValueAtTime(p_Args.time);
    int additiveKeyMode = 0;
    m_AdditiveKeyMode->getValueAtTime(p_Args.time, additiveKeyMode);
    params.additiveKeyMode = additiveKeyMode;
    params.additiveKeySaturation = static_cast<float>(m_AdditiveKeySaturation->getValueAtTime(p_Args.time));
    params.additiveKeyAmount = static_cast<float>(m_AdditiveKeyAmount->getValueAtTime(p_Args.time));
    params.additiveKeyBlackClamp = m_AdditiveKeyBlackClamp->getValueAtTime(p_Args.time);
    int viewMode = 0;
    m_ViewMode->getValueAtTime(p_Args.time, viewMode);
    params.viewMode = viewMode;

    IBKeyerCore::RenderRequest request;
    request.srcImage = src.get();
    request.screenImage = screen.get();
    request.backgroundImage = background.get();
    request.garbageMatteImage = garbageMatte.get();
    request.occlusionMatteImage = occlusionMatte.get();
    request.dstImage = dst.get();
    request.renderWindow = p_Args.renderWindow;
    request.hostCudaEnabled = p_Args.isEnabledCudaRender;
    request.hostCudaStream = p_Args.pCudaStream;
    request.hostMetalEnabled = p_Args.isEnabledMetalRender;
    request.hostMetalCmdQ = p_Args.pMetalCmdQ;
    request.params = params;

    // Hand the prepared request to the backend layer. 
    const IBKeyerCore::BackendResult result = IBKeyerCore::render(request);
    if (!result.success) {
        OFX::Log::print("IBKeyer: render failed. %s\n", result.detail.c_str());
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
}

bool IBKeyerPlugin::isIdentity(const OFX::IsIdentityArguments&,
                               OFX::Clip*&,
                               double&)
{
    // No identity case for a keyer — it always modifies alpha and usually RGB as well.
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////
// Moved from: the old "PARAMETER CHANGE HANDLER" section almost unchanged.
//
// 
// Enabled/disabled UI state is part of the OFX instance contract, not image processing.
void IBKeyerPlugin::changedParam(const OFX::InstanceChangedArgs&,
                                 const std::string& p_ParamName)
{
    if (p_ParamName == "useScreenInput" ||
        p_ParamName == "prematteEnabled" ||
        p_ParamName == "guidedFilterEnabled" ||
        p_ParamName == "additiveKeyEnabled" ||
        p_ParamName == "additiveKeyMode" ||
        p_ParamName == "nearGreyExtract" ||
        p_ParamName == "bgWrapEnabled") {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER
////////////////////////////////////////////////////////////////////////////////
// Moved from: the old "CLIP CHANGE HANDLER" section almost unchanged.
//
// 
// Clip connections affect which controls make sense to expose, so this stays in the plugin glue.
void IBKeyerPlugin::changedClip(const OFX::InstanceChangedArgs&,
                                const std::string& p_ClipName)
{
    if (p_ClipName == "Screen" ||
        p_ClipName == "Background" ||
        p_ClipName == "GarbageMatte" ||
        p_ClipName == "OcclusionMatte" ||
        p_ClipName == kOfxImageEffectSimpleSourceClipName) {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT
////////////////////////////////////////////////////////////////////////////////
// Moved from: the old "UI CONTROL ENABLEMENT" section.
//
// 
// Keeping it out of the backend avoids the mistake of mixing UI
// logic with frame processing logic.
void IBKeyerPlugin::setEnabledness()
{
    const bool useScreenInput = m_UseScreenInput->getValue();
    m_PickColor->setEnabled(!useScreenInput || !m_ScreenClip->isConnected());

    const bool prematteEnabled = m_PrematteEnabled->getValue();
    m_PrematteBlur->setEnabled(prematteEnabled);
    m_PrematteErode->setEnabled(prematteEnabled);
    m_PrematteIterations->setEnabled(prematteEnabled);

    const bool guidedEnabled = m_GuidedFilterEnabled->getValue();
    m_GuidedFilterMode->setEnabled(guidedEnabled);
    m_GuidedRadius->setEnabled(guidedEnabled);
    m_GuidedEpsilon->setEnabled(guidedEnabled);
    m_GuidedMix->setEnabled(guidedEnabled);
    m_EdgeProtect->setEnabled(guidedEnabled);
    m_RefineIterations->setEnabled(guidedEnabled);
    m_EdgeColorCorrect->setEnabled(guidedEnabled);

    const bool nearGreyEnabled = m_NearGreyExtract->getValue();
    m_NearGreyAmount->setEnabled(nearGreyEnabled);
    m_NearGreySoftness->setEnabled(nearGreyEnabled);

    // Resolve can be a little awkward about when optional secondary-input connection state becomes
    // visible to the plugin UI. If we require "checked + connected" here, users can end up in a
    // dead-feeling state where they enabled Background Wrap but still cannot edit its controls.
    //
    // Render-time validation still requires a real Background clip, so loosening the UI gate here
    // is a usability fix rather than a behavior change.
    const bool bgWrapControlsEnabled = m_BgWrapEnabled->getValue() ||
                                       (m_BgClip != nullptr && m_BgClip->isConnected());
    m_BgWrapBlur->setEnabled(bgWrapControlsEnabled);
    m_BgWrapAmount->setEnabled(bgWrapControlsEnabled);

    const bool additiveEnabled = m_AdditiveKeyEnabled->getValue();
    m_AdditiveKeyMode->setEnabled(additiveEnabled);
    m_AdditiveKeySaturation->setEnabled(additiveEnabled);
    m_AdditiveKeyAmount->setEnabled(additiveEnabled);
    m_AdditiveKeyBlackClamp->setEnabled(additiveEnabled);
}

OFX::DoubleParamDescriptor* defineDoubleParam(OFX::ImageEffectDescriptor& p_Desc,
                                              const std::string& p_Name,
                                              const std::string& p_Label,
                                              const std::string& p_Hint,
                                              OFX::GroupParamDescriptor* p_Parent,
                                              double p_DefaultValue,
                                              double p_MinValue,
                                              double p_MaxValue,
                                              double p_Increment)
{
    OFX::DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_DefaultValue);
    param->setRange(p_MinValue, p_MaxValue);
    param->setIncrement(p_Increment);
    param->setDisplayRange(p_MinValue, p_MaxValue);
    param->setDoubleType(OFX::eDoubleTypePlain);
    if (p_Parent != nullptr) {
        param->setParent(*p_Parent);
    }
    return param;
}

} // namespace

using namespace OFX;

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY
////////////////////////////////////////////////////////////////////////////////

IBKeyerFactory::IBKeyerFactory()
    : PluginFactoryHelper<IBKeyerFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void IBKeyerFactory::describe(ImageEffectDescriptor& p_Desc)
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

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
    // OpenDRT taught me an important rule here: the descriptor and the runtime policy have to tell
    // the same story. If describe() says "host CUDA exists" but render() quietly prefers staged CUDA
    // (or the other way around), Resolve can make bad assumptions during scan or playback.
    //
    // Because of that I use the shared selector from IBKeyerBackend.cpp for both places.
    // Default policy is host-preferred, while env overrides can still force INTERNAL if we ever need
    // to debug host interop separately from the CUDA algorithm itself.
    const bool advertiseHostCuda =
        (IBKeyerCore::selectedCudaRenderMode() == IBKeyerCore::CudaRenderMode::HostPreferred);
    p_Desc.setSupportsCudaRender(advertiseHostCuda);
    p_Desc.setSupportsCudaStream(advertiseHostCuda);
#elif defined(__APPLE__)
    // We only advertise Metal where we still have a real host-Metal implementation.
    // Windows/Linux now have a host-CUDA path, while macOS still keeps the older Metal route.
    p_Desc.setSupportsMetalRender(true);
#endif

    // Guided filtering is neighborhood-based, so claiming "no spatial awareness"
    // would invite incorrect ROI assumptions from hosts.
    p_Desc.setNoSpatialAwareness(false);
}

////////////////////////////////////////////////////////////////////////////////
// DESCRIBE IN CONTEXT — CLIPS + PARAMETERS
////////////////////////////////////////////////////////////////////////////////
// Moved from: the old "DESCRIBE IN CONTEXT — CLIPS + PARAMETERS" section.
//
// 
// Descriptor setup is part of the OFX host interface. It belongs near the factory, even after the
// processing code was split out, because the host needs this metadata before any backend exists.
void IBKeyerFactory::describeInContext(ImageEffectDescriptor& p_Desc, ContextEnum)
{
    // Source clip (foreground).
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Screen clip (optional clean plate).
    ClipDescriptor* screenClip = p_Desc.defineClip("Screen");
    screenClip->addSupportedComponent(ePixelComponentRGBA);
    screenClip->addSupportedComponent(ePixelComponentRGB);
    screenClip->setTemporalClipAccess(false);
    screenClip->setSupportsTiles(kSupportsTiles);
    screenClip->setOptional(true);
    screenClip->setIsMask(false);

    // Background clip (optional, used for background/light wrap parity with the older plugin).
    ClipDescriptor* bgClip = p_Desc.defineClip("Background");
    bgClip->addSupportedComponent(ePixelComponentRGBA);
    bgClip->addSupportedComponent(ePixelComponentRGB);
    bgClip->setTemporalClipAccess(false);
    bgClip->setSupportsTiles(kSupportsTiles);
    bgClip->setOptional(true);
    bgClip->setIsMask(false);

    ClipDescriptor* garbageClip = p_Desc.defineClip("GarbageMatte");
    garbageClip->addSupportedComponent(ePixelComponentRGBA);
    garbageClip->addSupportedComponent(ePixelComponentRGB);
    garbageClip->setTemporalClipAccess(false);
    garbageClip->setSupportsTiles(kSupportsTiles);
    garbageClip->setOptional(true);
    // These are semantically mattes, but treating them as normal optional image clips keeps the
    // host contract boring. That is useful on Resolve/Windows, where startup stability matters more
    // than hinting extra semantics that the effect does not strictly need.
    garbageClip->setIsMask(false);

    ClipDescriptor* occlusionClip = p_Desc.defineClip("OcclusionMatte");
    occlusionClip->addSupportedComponent(ePixelComponentRGBA);
    occlusionClip->addSupportedComponent(ePixelComponentRGB);
    occlusionClip->setTemporalClipAccess(false);
    occlusionClip->setSupportsTiles(kSupportsTiles);
    occlusionClip->setOptional(true);
    occlusionClip->setIsMask(false);

    // Output clip.
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    GroupParamDescriptor* screenGroup = p_Desc.defineGroupParam("ScreenGroup");
    screenGroup->setHint("Screen and keying parameters");
    screenGroup->setLabels("Screen Settings", "Screen Settings", "Screen Settings");

    ChoiceParamDescriptor* screenColor = p_Desc.defineChoiceParam("screenColor");
    screenColor->setLabel("Screen Color");
    screenColor->setHint("Dominant chroma of the backing screen.");
    screenColor->appendOption("Red");
    screenColor->appendOption("Green");
    screenColor->appendOption("Blue");
    screenColor->setDefault(IBKeyerCore::kScreenBlue);
    screenColor->setAnimates(true);
    screenColor->setParent(*screenGroup);
    page->addChild(*screenColor);

    BooleanParamDescriptor* useScreenInput = p_Desc.defineBooleanParam("useScreenInput");
    useScreenInput->setDefault(true);
    useScreenInput->setHint("When enabled, reads screen colour from the Screen clip. When disabled, uses the Pick Color constant.");
    useScreenInput->setLabels("Use Screen Input", "Use Screen Input", "Use Screen Input");
    useScreenInput->setParent(*screenGroup);
    page->addChild(*useScreenInput);

    RGBParamDescriptor* pickColor = p_Desc.defineRGBParam("pickColor");
    pickColor->setLabels("Pick Color", "Pick Color", "Pick Color");
    pickColor->setHint("Constant screen colour when Screen input is not connected.");
    pickColor->setDefault(0.0, 0.0, 0.0);
    pickColor->setParent(*screenGroup);
    page->addChild(*pickColor);

    GroupParamDescriptor* keyerGroup = p_Desc.defineGroupParam("KeyerGroup");
    keyerGroup->setHint("Keying and despill controls");
    keyerGroup->setLabels("Keyer Controls", "Keyer Controls", "Keyer Controls");

    page->addChild(*defineDoubleParam(p_Desc, "bias", "Bias",
                                      "Weighting between the two complement channels. 0.5 = equal weight.",
                                      keyerGroup, 0.5, 0.0, 1.0, 0.01));
    page->addChild(*defineDoubleParam(p_Desc, "limit", "Limit",
                                      "Scales the despill subtraction. 1.0 = standard.",
                                      keyerGroup, 1.0, 0.0, 5.0, 0.01));

    RGBParamDescriptor* respillColor = p_Desc.defineRGBParam("respillColor");
    respillColor->setLabels("Respill Color", "Respill Color", "Respill Color");
    respillColor->setHint("Colour to add back where screen spill was removed.");
    respillColor->setDefault(0.0, 0.0, 0.0);
    respillColor->setParent(*keyerGroup);
    page->addChild(*respillColor);

    BooleanParamDescriptor* premultiply = p_Desc.defineBooleanParam("premultiply");
    premultiply->setDefault(false);
    premultiply->setHint("Premultiply RGB by alpha for compositing.");
    premultiply->setLabels("Premultiply", "Premultiply", "Premultiply");
    premultiply->setParent(*keyerGroup);
    page->addChild(*premultiply);

    GroupParamDescriptor* matteGroup = p_Desc.defineGroupParam("MatteGroup");
    matteGroup->setHint("Matte refinement controls — adjust black/white points of the raw key");
    matteGroup->setLabels("Matte Controls", "Matte Controls", "Matte Controls");

    page->addChild(*defineDoubleParam(p_Desc, "blackClip", "Black Clip",
                                      "Crush blacks in the raw matte. Values below this become fully transparent. Useful for cleaning up noise in the screen area.",
                                      matteGroup, 0.0, 0.0, 1.0, 0.001));
    page->addChild(*defineDoubleParam(p_Desc, "whiteClip", "White Clip",
                                      "Push whites in the raw matte. Values above this become fully opaque. Useful for solidifying the foreground core.",
                                      matteGroup, 1.0, 0.0, 1.0, 0.001));
    page->addChild(*defineDoubleParam(p_Desc, "matteGamma", "Matte Gamma",
                                      "Applies a power curve to the alpha after black/white clipping.\n"
                                      "Values < 1.0 push semi-transparent edges toward opaque.\n"
                                      "Values > 1.0 push them toward transparent.\n"
                                      "1.0 = no change.",
                                      matteGroup, 1.0, 0.1, 4.0, 0.01));

    GroupParamDescriptor* prematteGroup = p_Desc.defineGroupParam("PrematteGroup");
    prematteGroup->setHint("Synthetic clean-plate generation used to re-key difficult shots.");
    prematteGroup->setLabels("Prematte", "Prematte", "Prematte");

    BooleanParamDescriptor* prematteEnabled = p_Desc.defineBooleanParam("prematteEnabled");
    prematteEnabled->setDefault(false);
    prematteEnabled->setHint("Builds a synthetic clean plate from the source and re-runs the core keyer. This moved out of the private Metal branch because it changes the actual key, not just the display.");
    prematteEnabled->setLabels("Enable", "Enable", "Enable");
    prematteEnabled->setParent(*prematteGroup);
    page->addChild(*prematteEnabled);

    IntParamDescriptor* prematteBlur = p_Desc.defineIntParam("prematteBlur");
    prematteBlur->setLabels("Blur Radius", "Blur Radius", "Blur Radius");
    prematteBlur->setScriptName("prematteBlur");
    prematteBlur->setHint("Blur radius for the synthetic clean plate.");
    prematteBlur->setDefault(8);
    prematteBlur->setRange(1, 200);
    prematteBlur->setDisplayRange(1, 50);
    prematteBlur->setParent(*prematteGroup);
    page->addChild(*prematteBlur);

    IntParamDescriptor* prematteErode = p_Desc.defineIntParam("prematteErode");
    prematteErode->setLabels("Erode", "Erode", "Erode");
    prematteErode->setScriptName("prematteErode");
    prematteErode->setHint("Erodes the initial matte before clean-plate estimation to reduce foreground contamination.");
    prematteErode->setDefault(0);
    prematteErode->setRange(0, 20);
    prematteErode->setDisplayRange(0, 10);
    prematteErode->setParent(*prematteGroup);
    page->addChild(*prematteErode);

    IntParamDescriptor* prematteIterations = p_Desc.defineIntParam("prematteIterations");
    prematteIterations->setLabels("Iterations", "Iterations", "Iterations");
    prematteIterations->setScriptName("prematteIterations");
    prematteIterations->setHint("How many times the synthetic clean plate is rebuilt and re-keyed.");
    prematteIterations->setDefault(1);
    prematteIterations->setRange(1, 5);
    prematteIterations->setDisplayRange(1, 5);
    prematteIterations->setParent(*prematteGroup);
    page->addChild(*prematteIterations);

    GroupParamDescriptor* ngeGroup = p_Desc.defineGroupParam("NGEGroup");
    ngeGroup->setHint("Near Grey Extraction controls");
    ngeGroup->setLabels("Near Grey Extract", "Near Grey Extract", "Near Grey Extract");

    BooleanParamDescriptor* nearGreyExtract = p_Desc.defineBooleanParam("nearGreyExtract");
    nearGreyExtract->setDefault(true);
    nearGreyExtract->setHint("Improves matte quality in near-grey or ambiguous areas.");
    nearGreyExtract->setLabels("Enable", "Enable", "Enable");
    nearGreyExtract->setParent(*ngeGroup);
    page->addChild(*nearGreyExtract);

    page->addChild(*defineDoubleParam(p_Desc, "nearGreyAmount", "Strength",
                                      "How much the near-grey extraction contributes to the final alpha.",
                                      ngeGroup, 0.5, 0.0, 1.0, 0.01));
    page->addChild(*defineDoubleParam(p_Desc, "nearGreySoftness", "Softness",
                                      "Controls how the keyer measures 'greyness' in ambiguous regions.",
                                      ngeGroup, 1.0, 0.0, 1.0, 0.01));

    GroupParamDescriptor* guidedGroup = p_Desc.defineGroupParam("GuidedFilterGroup");
    guidedGroup->setHint("Edge-aware matte refinement using the source luminance as guide");
    guidedGroup->setLabels("Guided Filter", "Guided Filter", "Guided Filter");

    BooleanParamDescriptor* guidedEnabled = p_Desc.defineBooleanParam("guidedFilterEnabled");
    // This intentionally matches the older IBKeymaster defaults now. Earlier in the port I turned
    // this off to stay closer to the simpler Gaffer graph, but once the goal shifted to full
    // IBKeymaster parity that default became misleading.
    guidedEnabled->setDefault(true);
    guidedEnabled->setHint("Enable guided filter matte refinement. Uses source luminance as an edge guide to recover hair detail and soft edges.");
    guidedEnabled->setLabels("Enable", "Enable", "Enable");
    guidedEnabled->setParent(*guidedGroup);
    page->addChild(*guidedEnabled);

    ChoiceParamDescriptor* guidedMode = p_Desc.defineChoiceParam("guidedFilterMode");
    guidedMode->setLabel("Guide Mode");
    guidedMode->setHint("Luma uses the simpler scalar guide. RGB uses the full 3-channel guided filter from the private Metal branch.");
    guidedMode->appendOption("Luma");
    guidedMode->appendOption("RGB");
    guidedMode->setDefault(0);
    guidedMode->setAnimates(true);
    guidedMode->setParent(*guidedGroup);
    page->addChild(*guidedMode);

    IntParamDescriptor* guidedRadius = p_Desc.defineIntParam("guidedRadius");
    guidedRadius->setLabels("Radius", "Radius", "Radius");
    guidedRadius->setScriptName("guidedRadius");
    guidedRadius->setHint("Filter window radius in pixels.");
    guidedRadius->setDefault(8);
    guidedRadius->setRange(1, 100);
    guidedRadius->setDisplayRange(1, 50);
    guidedRadius->setParent(*guidedGroup);
    page->addChild(*guidedRadius);

    page->addChild(*defineDoubleParam(p_Desc, "guidedEpsilon", "Epsilon",
                                      "Edge sensitivity. Smaller values preserve more edges but may introduce noise.",
                                      guidedGroup, 0.01, 0.0001, 1.0, 0.001));
    page->addChild(*defineDoubleParam(p_Desc, "guidedMix", "Mix",
                                      "Blend between raw matte (0.0) and guided-filter-refined matte (1.0).",
                                      guidedGroup, 1.0, 0.0, 1.0, 0.01));
    page->addChild(*defineDoubleParam(p_Desc, "edgeProtect", "Edge Protection",
                                      "Blends the guide signal from source luminance toward the raw alpha.",
                                      guidedGroup, 0.5, 0.0, 1.0, 0.01));

    IntParamDescriptor* refineIterations = p_Desc.defineIntParam("refineIterations");
    refineIterations->setLabels("Refine Iterations", "Refine Iterations", "Refine Iterations");
    refineIterations->setScriptName("refineIterations");
    refineIterations->setHint("Number of iterative guided-filter refinement passes.");
    refineIterations->setDefault(2);
    refineIterations->setRange(1, 5);
    refineIterations->setDisplayRange(1, 5);
    refineIterations->setParent(*guidedGroup);
    page->addChild(*refineIterations);

    page->addChild(*defineDoubleParam(p_Desc, "edgeColorCorrect", "Edge Color Correct",
                                      "Re-estimates foreground colour at semi-transparent edges using the matting equation.",
                                      guidedGroup, 0.0, 0.0, 1.0, 0.01));

    GroupParamDescriptor* bgGroup = p_Desc.defineGroupParam("BgWrapGroup");
    bgGroup->setHint("Bleeds a blurred version of the new background into the foreground edges.");
    bgGroup->setLabels("Background Wrap", "Background Wrap", "Background Wrap");

    BooleanParamDescriptor* bgWrapEnabled = p_Desc.defineBooleanParam("bgWrapEnabled");
    bgWrapEnabled->setDefault(false);
    bgWrapEnabled->setHint("Enable background wrap. Requires the Background clip to be connected.");
    bgWrapEnabled->setLabels("Enable", "Enable", "Enable");
    bgWrapEnabled->setParent(*bgGroup);
    page->addChild(*bgWrapEnabled);

    IntParamDescriptor* bgWrapBlur = p_Desc.defineIntParam("bgWrapBlur");
    bgWrapBlur->setLabels("Blur Radius", "Blur Radius", "Blur Radius");
    bgWrapBlur->setScriptName("bgWrapBlur");
    bgWrapBlur->setHint("Gaussian blur radius applied to the background before wrapping.");
    bgWrapBlur->setDefault(20);
    bgWrapBlur->setRange(1, 200);
    bgWrapBlur->setDisplayRange(1, 100);
    bgWrapBlur->setParent(*bgGroup);
    page->addChild(*bgWrapBlur);

    page->addChild(*defineDoubleParam(p_Desc, "bgWrapAmount", "Amount",
                                      "How much blurred background to bleed into the foreground edges.",
                                      bgGroup, 0.5, 0.0, 2.0, 0.01));

    GroupParamDescriptor* additiveGroup = p_Desc.defineGroupParam("AdditiveKeyGroup");
    additiveGroup->setHint("Recovers transparent detail the alpha missed by adding back source-minus-screen detail.");
    additiveGroup->setLabels("Additive Key", "Additive Key", "Additive Key");

    BooleanParamDescriptor* additiveEnabled = p_Desc.defineBooleanParam("additiveKeyEnabled");
    additiveEnabled->setDefault(false);
    additiveEnabled->setHint("Enable additive detail recovery.");
    additiveEnabled->setLabels("Enable", "Enable", "Enable");
    additiveEnabled->setParent(*additiveGroup);
    page->addChild(*additiveEnabled);

    ChoiceParamDescriptor* additiveMode = p_Desc.defineChoiceParam("additiveKeyMode");
    additiveMode->setLabel("Mode");
    additiveMode->setHint("Addition uses source minus screen directly. Multiply uses a factor against the blurred background.");
    additiveMode->appendOption("Addition");
    additiveMode->appendOption("Multiply");
    additiveMode->setDefault(0);
    additiveMode->setAnimates(true);
    additiveMode->setParent(*additiveGroup);
    page->addChild(*additiveMode);

    page->addChild(*defineDoubleParam(p_Desc, "additiveKeySaturation", "Saturation",
                                      "Desaturates recovered detail to reduce residual screen colour.",
                                      additiveGroup, 0.0, 0.0, 1.0, 0.01));
    page->addChild(*defineDoubleParam(p_Desc, "additiveKeyAmount", "Amount",
                                      "Strength of the additive detail recovery.",
                                      additiveGroup, 0.0, 0.0, 2.0, 0.01));

    BooleanParamDescriptor* additiveClamp = p_Desc.defineBooleanParam("additiveKeyBlackClamp");
    additiveClamp->setDefault(false);
    additiveClamp->setHint("Clamp additive detail to positive values only.");
    additiveClamp->setLabels("Black Clamp", "Black Clamp", "Black Clamp");
    additiveClamp->setParent(*additiveGroup);
    page->addChild(*additiveClamp);

    GroupParamDescriptor* displayGroup = p_Desc.defineGroupParam("DisplayGroup");
    displayGroup->setHint("Diagnostic views from the richer private branch.");
    displayGroup->setLabels("Display", "Display", "Display");

    ChoiceParamDescriptor* viewMode = p_Desc.defineChoiceParam("viewMode");
    viewMode->setLabel("View Mode");
    viewMode->setHint("Displays intermediate pipeline stages for debugging and tuning.");
    viewMode->appendOption("Composite");
    viewMode->appendOption("Raw Matte");
    viewMode->appendOption("Clean Plate");
    viewMode->appendOption("Refined Matte");
    viewMode->appendOption("Despilled Source");
    viewMode->appendOption("Blurred Background");
    viewMode->setDefault(0);
    viewMode->setAnimates(true);
    viewMode->setParent(*displayGroup);
    page->addChild(*viewMode);
}

////////////////////////////////////////////////////////////////////////////////
// CREATE INSTANCE
////////////////////////////////////////////////////////////////////////////////

ImageEffect* IBKeyerFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum)
{
    return new IBKeyerPlugin(p_Handle);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION
////////////////////////////////////////////////////////////////////////////////

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static IBKeyerFactory ibKeyer;
    p_FactoryArray.push_back(&ibKeyer);
}
