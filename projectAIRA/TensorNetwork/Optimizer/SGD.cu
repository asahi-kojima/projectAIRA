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

			void SGD::optimize_unique()
			{

				for (auto iter = m_OptimizeScheduled_BaseLayer_tbl.begin(), end = m_OptimizeScheduled_BaseLayer_tbl.end(); iter != end; iter++)
				{
					if (const std::shared_ptr<layer::BaseLayer>& pLayercore = (*iter).lock())
					{
						const auto& baseLayer = *pLayercore;
						bool on_cuda = baseLayer.isOnCuda();
						for (auto iter = baseLayer.getTrainableParamTbl().begin(), end = baseLayer.getTrainableParamTbl().end(); iter != end; iter++)
						{
							auto& parameter_tensor = *(*iter);
							if (!parameter_tensor.requiresGrad())
							{
								continue;
							}

							const u32 dataSize = parameter_tensor.getDataSize();

							if (on_cuda)
							{
								dim3 block(32);
								dim3 grid((dataSize + block.x - 1) / block.x);
								optimize_on_gpu << <grid, block >> > (
									parameter_tensor.getGpuDataAddress(),
									parameter_tensor.getGpuGradDataAddress(),
									mLearningRate,
									dataSize);
								CUDA_SYNCHRONIZE_DEBUG;
							}
							else
							{
								for (u32 i = 0; i < dataSize; i++)
								{
									parameter_tensor(i) -= mLearningRate * parameter_tensor.d(i);
								}
							}
						}
					}
				}
			}


		}
	}
}