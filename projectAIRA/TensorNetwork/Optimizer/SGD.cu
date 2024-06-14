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

using SGD = aoba::nn::optimizer::Optimizer::SGD;

SGD::SGD(DataType learningRate)
	: OptimizerSkeleton(learningRate)
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

	for (auto iter = m_OptimizeScheduled_LayerCore_tbl.begin(), end = m_OptimizeScheduled_LayerCore_tbl.end(); iter != end; iter++)
	{
		if (const std::shared_ptr<layer::Layer::LayerSkeleton>& pLayercore = (*iter).lock())
		{
			const auto& layercore = *pLayercore;
			bool on_cuda = layercore.m_on_cuda;
			for (auto iter = layercore.m_parameter_tbl.begin(), end = layercore.m_parameter_tbl.end(); iter != end; iter++)
			{
				auto& parameter_tensor = *(*iter);
				if (!parameter_tensor._m_need_grad)
				{
					continue;
				}

				const u32 dataSize = parameter_tensor.mDataSize;
				
				if (on_cuda)
				{
					dim3 block(32);
					dim3 grid((dataSize + block.x - 1) / block.x);
					optimize_on_gpu<<<grid, block>>>(
						parameter_tensor._m_gpu_data_address, 
						parameter_tensor._m_gpu_grad_data_address, 
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

