// EPD-8 compiled execution-backend capability probe.
// This reports build capabilities only; it does not claim that a GPU is
// physically present or that the NVIDIA container runtime is healthy.

#include <iostream>

#ifndef EPD_ENABLE_TENSORRT
#define EPD_ENABLE_TENSORRT 0
#endif

#ifndef USE_GPU
#define USE_GPU false
#endif

int main()
{
#if defined(__aarch64__)
  constexpr const char * architecture = "aarch64";
#elif defined(__x86_64__)
  constexpr const char * architecture = "x86_64";
#else
  constexpr const char * architecture = "other";
#endif

  std::cout
    << "{\"schema_version\":\"epd_backend_probe/v1\","
    << "\"architecture\":\"" << architecture << "\","
    << "\"cpu\":true,"
    << "\"cuda\":" << (USE_GPU ? "true" : "false") << ","
    << "\"tensorrt\":"
    << ((USE_GPU && EPD_ENABLE_TENSORRT) ? "true" : "false")
    << "}" << std::endl;
  return 0;
}
