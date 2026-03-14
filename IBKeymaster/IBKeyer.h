#pragma once

#include "ofxsImageEffect.h"

// The factory stays tiny on purpose. The interesting work now lives in
// backend/request code so the OFX entry points remain readable.
class IBKeyerFactory : public OFX::PluginFactoryHelper<IBKeyerFactory>
{
public:
    IBKeyerFactory();
    virtual void load() {}
    virtual void unload() {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};
