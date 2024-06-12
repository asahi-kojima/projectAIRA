#include "SGD.h"
#include "Layer/Layer.h"
using SGD = aoba::nn::optimizer::Optimizer::SGD;


SGD::SGD(u32 learningRate)
	: OptimizerSkeleton(learningRate)
{

}

SGD::~SGD() {}

void SGD::initialize()
{
#ifdef _DEBUG
	std::cout << "SGD initialize" << std::endl;
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

