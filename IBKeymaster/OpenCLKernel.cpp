// IBKeymaster v2 OpenCL Kernel — Stub
// Falls back to CPU processing

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif
#include <cstring>
#include <cmath>
#include <stdio.h>

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
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
                     const float* p_Background, float* p_Output)
{
    // Stub — OpenCL not implemented, CPU fallback handles processing
}
