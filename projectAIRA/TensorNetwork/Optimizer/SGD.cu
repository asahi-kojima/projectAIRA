#include "Layer/Layer.h"
#include "SGD.h"

namespace
{
	__global__ void optimize_on_gpu(DataType* param, DataType* dParam, DataType learningRate, u32 dataSize)
	{
		u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		DataType value = param[index];
		param[index] = value - dParam[index] * learningRate;
		//printf("%f : %f : %f\n",value , param[index], dParam[index]);
	}
}


namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{

			SGD::SGD(DataType learningRate)
				: BaseOptimizer(learningRate)
			{

			}

			SGD::~SGD() {}

			void SGD::initialize()
			{
#ifdef _DEBUG
				std::cout << "SGD::initialize()" << std::endl;
#endif
			}

			void SGD::optimize_unique(tensor::TensorCore& parameter)
			{			
				const u32 paramSize = parameter.getDataSize();
				if (parameter.isOnCuda())
				{
					dim3 block(32);
					dim3 grid((paramSize + block.x - 1) / block.x);
					optimize_on_gpu << <grid, block >> > (
						parameter.getGpuDataAddress(),
						parameter.getGpuGradDataAddress(),
						mLearningRate,
						paramSize);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					for (u32 i = 0; i < paramSize; i++)
					{
						parameter(i) -= mLearningRate * parameter.d(i);
					}
				}
				
			}


		}
	}
}