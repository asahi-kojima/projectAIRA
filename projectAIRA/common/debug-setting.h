#pragma once
#include "typeinfo.h"

#define ON 1
#define OFF 0

#ifdef _DEBUG
#define DEBUG_MODE 1
#else
#define DEBUG_MODE 0
#endif

#define GPU_SYNC_DEBUG (ON & _DEBUG)
#define CPU_DEBUG_ON (OFF & DEBUG_MODE)
#define INDEX_DEBUG (ON & DEBUG_MODE)


////Timeデバッグは以下のIndexデバッグと併用すると
////正確な値が出ないので注意。
////GPU使用時にGPUSyncが切れているとTimeデバッグは
////正確な値が出ないので注意。
#if _DEBUG
#define TIME_DEBUG (ON & _DEBUG)
#endif
#if TIME_DEBUG
#include <map>
#include <string>
#include <chrono>
namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			extern std::map<std::string, f32> debugTimers;
		}
	}
}

inline std::string makeDebugIdentifier(const u32 id, const char* functionName, const char* detailFunctionName)
{
    const u32 IdLength = 4;
    std::string id_as_string = std::to_string(id);
    const u32 complementSize =( (0 >= (IdLength - id_as_string.length())) ? 0 : IdLength - id_as_string.length() );
    std::string name(complementSize, '0');
    name += id_as_string += " : ";
    name += (std::string(functionName) += detailFunctionName);

    return name;
}
#endif


#ifdef _DEBUG
#define CUDA_SYNCHRONIZE_DEBUG CHECK(cudaDeviceSynchronize())
#else
#define CUDA_SYNCHRONIZE_DEBUG {}
#endif

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
