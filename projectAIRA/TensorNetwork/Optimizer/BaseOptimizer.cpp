#include "BaseOptimizer.h"


namespace aoba::nn::optimizer
{

	BaseOptimizer::BaseOptimizer(DataType learningRate)
		:mLearningRate(learningRate)
		//, m_OptimizeScheduled_BaseLayer_tbl(0)
		, mOptimizeScheduledParamTbl(0)
		, mIsInitialized(false)
	{
	}

	BaseOptimizer::~BaseOptimizer()
	{
	}

	void BaseOptimizer::operator()(const layer::Layer& layer)
	{
		const auto& baseLayer = *layer.getBaseLayer();

		for (auto iter = baseLayer.getTrainableParamTbl().begin(), end = baseLayer.getTrainableParamTbl().end(); iter != end; iter++)
		{
			mOptimizeScheduledParamTbl.push_back(*iter);
		}

		for (auto iter = baseLayer.getInternalLayerTbl().begin(), end = baseLayer.getInternalLayerTbl().end(); iter != end; iter++)
		{
			(*this)(iter->second);
		}
	}

	void BaseOptimizer::optimize()
	{
		if (!mIsInitialized)
		{
			initialize();
			mIsInitialized = true;
		}

		for (auto iter = mOptimizeScheduledParamTbl.begin(), end = mOptimizeScheduledParamTbl.end(); iter != end; iter++)
		{
			if (const std::shared_ptr<tensor::TensorCore>& parameter = (*iter).lock())
			{
				optimize_unique(*parameter);
			}
			else
			{
				std::cout << "Resource Error@BaseOptimizer::optimize" << std::endl;
				exit(1);
			}
		}
	}

}
