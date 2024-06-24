#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <iostream>
#include "typeinfo.h"
#include "debug-setting.h"

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#define LINE_WIDTH 100

inline void informationFormat(std::string s)
{
#if 0  
	std::cout << "  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ " << std::endl;
	std::cout << " / / / / / / / / / / / / / / / / / / / / / / / / / / / / " << std::endl;
	std::string bar = " / / / / / / / / / / / / / / / / / / / / / / / / / / / / ";
	unsigned int sideLength = (bar.length() - s.length()) / 2;
	std::cout << std::string(sideLength, ' ') + s + std::string(sideLength, ' ') << std::endl;
	std::cout << "/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << std::endl;

#else	
	std::string topBar = "  ";
	std::string topBarParts = "_ ";
	for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
	{
		topBar = topBar + topBarParts;
	}
	std::cout << topBar << std::endl;

	std::string bar = "";
	std::string parts = " /";
	for (int i = 0; i < LINE_WIDTH / 2; i++)
	{
		bar = bar + parts;
	}
	std::cout << bar << std::endl;

	unsigned int sideLength = (bar.length() - s.length()) / 2;
	std::cout << std::string(sideLength, ' ') + s + std::string(sideLength, ' ') << std::endl;

	std::string underBar = "/";
	std::string underBarParts = "_/";
	for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
	{
		underBar = underBar + underBarParts;
	}
	std::cout << underBar << std::endl;
#endif
}

class gpu_manager
{
public:
	static bool gpu_is_available()
	{
		informationFormat("GPU Information");

		s32 gpuDeviceNum = 0;
		const cudaError_t error = cudaGetDeviceCount(&gpuDeviceNum);
		if (error != cudaSuccess)
		{
			std::cout << "Error : " << __FILE__ << ":" << __LINE__ << std::endl;
			std::cout << "code : " << error << std::endl;
			std::cout << "reason : " << cudaGetErrorString(error) << std::endl;
			std::cout << "\nSystem can't get any Device Information. So this AI uses CPU to DeepLearning.\n";
			return false;
		}

		if (gpuDeviceNum == 0)
		{
			std::cout << "No GPU Device that support CUDA. So this AI use CPU  to DeepLearning.";
			return false;
		}

		std::cout << gpuDeviceNum << " GPU device(s) that support CUDA detected.\n";

		s32 optimal_gpu_id = 0;
		s32 maxMultiProcessorCount = 0;

		for (s32 index = 0; index < gpuDeviceNum; index++)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, index);
			s32 driverVersion = 0;
			s32 runtimeVersion = 0;
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			auto formater = [](std::string s, s32 length = 50)
				{
					return s + std::string(std::max(0, length - static_cast<s32>(s.length())), ' ') + " : ";
				};

			std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
			std::cout << "Information of DeviceID = " << index << "\n" << std::endl;
			std::cout << formater("Device name") << "\"" << deviceProp.name << "\"" << std::endl;
			std::cout << formater("CUDA Driver Version") << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
			std::cout << formater("CUDA Runtime Versionz") << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
			std::cout << formater("CUDA Capability Major/Minor version number") << deviceProp.major << "." << deviceProp.minor << std::endl;

			std::cout << formater("VRAM") << static_cast<f32>(deviceProp.totalGlobalMem / pow(1024.0, 3)) << "GB (" << deviceProp.totalGlobalMem << "Bytes)" << std::endl;
			std::cout << formater("Total amount of shared memory per block") << deviceProp.sharedMemPerBlock << "Bytes" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 1D") << "(" << deviceProp.maxTexture1D << ")" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 2D") << "(" << deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture2D[1] << ")" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 3D") << "(" << deviceProp.maxTexture3D[0] << ", " << deviceProp.maxTexture3D[1] << ", " << deviceProp.maxTexture3D[2] << ")" << std::endl;
			std::cout << formater("Maximum sizes of threads per block") << deviceProp.maxThreadsPerBlock << std::endl;
			std::cout << formater("Maximum sizes of each dimension of a block") << "(" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
			std::cout << formater("Maximum sizes of each dimension of a grid") << "(" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;



			if (deviceProp.multiProcessorCount > maxMultiProcessorCount)
			{
				maxMultiProcessorCount = deviceProp.multiProcessorCount;
				optimal_gpu_id = index;
			}
		}

		std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << "\n" << std::endl;
		std::cout << "In this case, the " << optimal_gpu_id << "th GPU will be used" << std::endl;
		cudaSetDevice(optimal_gpu_id);
		std::cout << std::endl;
		return  CPU_DEBUG_ON ? false : true;
	}
};