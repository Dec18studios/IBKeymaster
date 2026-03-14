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
constexpr int kPluginVersionMinor = 1;
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

    OFX::ChoiceParam* m_ScreenColor;
    OFX::BooleanParam* m_UseScreenInput;
    OFX::RGBParam* m_PickColor;
    OFX::DoubleParam* m_Bias;
    OFX::DoubleParam* m_Limit;
    OFX::RGBParam* m_RespillColor;
    OFX::BooleanParam* m_Premultiply;
    OFX::DoubleParam* m_BlackClip;
    OFX::DoubleParam* m_WhiteClip;
    OFX::BooleanParam* m_NearGreyExtract;
    OFX::DoubleParam* m_NearGreyAmount;
    OFX::BooleanParam* m_GuidedFilterEnabled;
    OFX::IntParam* m_GuidedRadius;
    OFX::DoubleParam* m_GuidedEpsilon;
    OFX::DoubleParam* m_GuidedMix;
};

IBKeyerPlugin::IBKeyerPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
    , m_DstClip(fetchClip(kOfxImageEffectOutputClipName))
    , m_SrcClip(fetchClip(kOfxImageEffectSimpleSourceClipName))
    , m_ScreenClip(fetchClip("Screen"))
    , m_ScreenColor(fetchChoiceParam("screenColor"))
    , m_UseScreenInput(fetchBooleanParam("useScreenInput"))
    , m_PickColor(fetchRGBParam("pickColor"))
    , m_Bias(fetchDoubleParam("bias"))
    , m_Limit(fetchDoubleParam("limit"))
    , m_RespillColor(fetchRGBParam("respillColor"))
    , m_Premultiply(fetchBooleanParam("premultiply"))
    , m_BlackClip(fetchDoubleParam("blackClip"))
    , m_WhiteClip(fetchDoubleParam("whiteClip"))
    , m_NearGreyExtract(fetchBooleanParam("nearGreyExtract"))
    , m_NearGreyAmount(fetchDoubleParam("nearGreyAmount"))
    , m_GuidedFilterEnabled(fetchBooleanParam("guidedFilterEnabled"))
    , m_GuidedRadius(fetchIntParam("guidedRadius"))
    , m_GuidedEpsilon(fetchDoubleParam("guidedEpsilon"))
    , m_GuidedMix(fetchDoubleParam("guidedMix"))
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
    params.nearGreyExtract = m_NearGreyExtract->getValueAtTime(p_Args.time);
    params.nearGreyAmount = static_cast<float>(m_NearGreyAmount->getValueAtTime(p_Args.time));
    params.guidedFilterEnabled = m_GuidedFilterEnabled->getValueAtTime(p_Args.time);
    params.guidedRadius = m_GuidedRadius->getValueAtTime(p_Args.time);
    params.guidedEpsilon = static_cast<float>(m_GuidedEpsilon->getValueAtTime(p_Args.time));
    params.guidedMix = static_cast<float>(m_GuidedMix->getValueAtTime(p_Args.time));

    IBKeyerCore::RenderRequest request;
    request.srcImage = src.get();
    request.screenImage = screen.get();
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
        p_ParamName == "guidedFilterEnabled" ||
        p_ParamName == "nearGreyExtract") {
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
    if (p_ClipName == "Screen" || p_ClipName == kOfxImageEffectSimpleSourceClipName) {
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

    const bool guidedEnabled = m_GuidedFilterEnabled->getValue();
    m_GuidedRadius->setEnabled(guidedEnabled);
    m_GuidedEpsilon->setEnabled(guidedEnabled);
    m_GuidedMix->setEnabled(guidedEnabled);

    const bool nearGreyEnabled = m_NearGreyExtract->getValue();
    m_NearGreyAmount->setEnabled(nearGreyEnabled);
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
    screenClip->addSupportedComponent(ePixelComponentRGB);
    screenClip->addSupportedComponent(ePixelComponentRGBA);
    screenClip->setTemporalClipAccess(false);
    screenClip->setSupportsTiles(kSupportsTiles);
    screenClip->setOptional(true);
    screenClip->setIsMask(false);

    // Output clip.
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Page.
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Group: Screen Settings.
    GroupParamDescriptor* screenGroup = p_Desc.defineGroupParam("ScreenGroup");
    screenGroup->setHint("Screen and keying parameters");
    screenGroup->setLabels("Screen Settings", "Screen Settings", "Screen Settings");

    // Screen colour choice (Red / Green / Blue).
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

    // Use screen input toggle.
    BooleanParamDescriptor* useScreenInput = p_Desc.defineBooleanParam("useScreenInput");
    useScreenInput->setDefault(true);
    useScreenInput->setHint("When enabled, reads screen colour from the Screen clip. When disabled, uses the Pick Color constant.");
    useScreenInput->setLabels("Use Screen Input", "Use Screen Input", "Use Screen Input");
    useScreenInput->setParent(*screenGroup);
    page->addChild(*useScreenInput);

    // Pick colour (constant fallback).
    RGBParamDescriptor* pickColor = p_Desc.defineRGBParam("pickColor");
    pickColor->setLabels("Pick Color", "Pick Color", "Pick Color");
    pickColor->setHint("Constant screen colour when Screen input is not connected.");
    pickColor->setDefault(0.0, 0.0, 0.0);
    pickColor->setParent(*screenGroup);
    page->addChild(*pickColor);

    // Group: Keyer Controls.
    GroupParamDescriptor* keyerGroup = p_Desc.defineGroupParam("KeyerGroup");
    keyerGroup->setHint("Keying and despill controls");
    keyerGroup->setLabels("Keyer Controls", "Keyer Controls", "Keyer Controls");

    page->addChild(*defineDoubleParam(p_Desc, "bias", "Bias",
                                      "Weighting between the two complement channels. 0.5 = equal weight.",
                                      keyerGroup, 0.5, 0.0, 1.0, 0.01));
    page->addChild(*defineDoubleParam(p_Desc, "limit", "Limit",
                                      "Scales the despill subtraction. 1.0 = standard.",
                                      keyerGroup, 1.0, 0.0, 5.0, 0.01));

    // Respill colour.
    RGBParamDescriptor* respillColor = p_Desc.defineRGBParam("respillColor");
    respillColor->setLabels("Respill Color", "Respill Color", "Respill Color");
    respillColor->setHint("Colour to add back where screen spill was removed.");
    respillColor->setDefault(0.0, 0.0, 0.0);
    respillColor->setParent(*keyerGroup);
    page->addChild(*respillColor);

    // Premultiply.
    BooleanParamDescriptor* premultiply = p_Desc.defineBooleanParam("premultiply");
    premultiply->setDefault(false);
    premultiply->setHint("Premultiply RGB by alpha for compositing.");
    premultiply->setLabels("Premultiply", "Premultiply", "Premultiply");
    premultiply->setParent(*keyerGroup);
    page->addChild(*premultiply);

    // Group: Matte Controls.
    GroupParamDescriptor* matteGroup = p_Desc.defineGroupParam("MatteGroup");
    matteGroup->setHint("Matte refinement controls");
    matteGroup->setLabels("Matte Controls", "Matte Controls", "Matte Controls");

    page->addChild(*defineDoubleParam(p_Desc, "blackClip", "Black Clip",
                                      "Crush blacks in the raw matte. Values below this become fully transparent.",
                                      matteGroup, 0.0, 0.0, 1.0, 0.001));
    page->addChild(*defineDoubleParam(p_Desc, "whiteClip", "White Clip",
                                      "Push whites in the raw matte. Values above this become fully opaque.",
                                      matteGroup, 1.0, 0.0, 1.0, 0.001));

    // Group: Near Grey Extract.
    GroupParamDescriptor* nearGreyGroup = p_Desc.defineGroupParam("NGEGroup");
    nearGreyGroup->setHint("Near Grey Extraction controls");
    nearGreyGroup->setLabels("Near Grey Extract", "Near Grey Extract", "Near Grey Extract");

    // Near Grey Extract toggle.
    BooleanParamDescriptor* nearGreyExtract = p_Desc.defineBooleanParam("nearGreyExtract");
    nearGreyExtract->setDefault(true);
    nearGreyExtract->setHint("Improves matte quality in near-grey or ambiguous areas.");
    nearGreyExtract->setLabels("Enable", "Enable", "Enable");
    nearGreyExtract->setParent(*nearGreyGroup);
    page->addChild(*nearGreyExtract);

    // Near Grey Amount.
    page->addChild(*defineDoubleParam(p_Desc, "nearGreyAmount", "Amount",
                                      "Controls the near-grey response curve used by the keyer.",
                                      nearGreyGroup, 1.0, 0.0, 1.0, 0.01));

    // Group: Guided Filter.
    GroupParamDescriptor* guidedGroup = p_Desc.defineGroupParam("GuidedFilterGroup");
    guidedGroup->setHint("Edge-aware matte refinement using the source luminance as guide");
    guidedGroup->setLabels("Guided Filter", "Guided Filter", "Guided Filter");

    BooleanParamDescriptor* guidedEnabled = p_Desc.defineBooleanParam("guidedFilterEnabled");
    // The original Gaffer IBKeyer stops at the raw IBK-style result plus optional premultiply.
    // Guided filtering is useful, but it is an extension we added in the OFX port, not part of
    // the source graph itself. Defaulting it off keeps "fresh instance" behaviour closer to the
    // original tool and makes backend parity checks less confusing.
    guidedEnabled->setDefault(false);
    guidedEnabled->setHint("Enable guided filter matte refinement.");
    guidedEnabled->setLabels("Enable", "Enable", "Enable");
    guidedEnabled->setParent(*guidedGroup);
    page->addChild(*guidedEnabled);

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
                                      "Edge sensitivity for the guided filter.",
                                      guidedGroup, 0.01, 0.0001, 1.0, 0.001));
    page->addChild(*defineDoubleParam(p_Desc, "guidedMix", "Mix",
                                      "Blend between raw matte and guided-filter-refined matte.",
                                      guidedGroup, 1.0, 0.0, 1.0, 0.01));
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
