#pragma once
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
////Time�f�o�b�O�͈ȉ���Index�f�o�b�O�ƕ��p�����
////���m�Ȓl���o�Ȃ��̂Œ��ӁB
////GPU�g�p����GPUSync���؂�Ă����Time�f�o�b�O��
////���m�Ȓl���o�Ȃ��̂Œ��ӁB
//#define TIME_DEBUG (ON & _DEBUG)
//
//#if TIME_DEBUG
//#include <map>
//#include <string>
//#include <chrono>
//extern std::map<std::string, f32> timers;
//#endif


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