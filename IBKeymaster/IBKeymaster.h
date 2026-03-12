#pragma once

#include "ofxsImageEffect.h"

////////////////////////////////////////////////////////////////////////////////
// IBKeymaster OFX Plugin Factory
// Port of Jed Smith's IBKeymaster from gaffer-tools
////////////////////////////////////////////////////////////////////////////////
class IBKeymasterFactory : public OFX::PluginFactoryHelper<IBKeymasterFactory>
{
public:
    IBKeymasterFactory();
    virtual void load() {}
    virtual void unload() {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};
