#include "BaseOptimizer.h"


namespace aoba::nn::optimizer
{

	BaseOptimizer::BaseOptimizer(DataType learningRate)
		:mLearningRate(learningRate)
		, m_OptimizeScheduled_BaseLayer_tbl(0)
		, mIsInitialized(false)
	{
	}

	BaseOptimizer::~BaseOptimizer()
	{
	}

	void BaseOptimizer::operator()(const layer::Layer& layer)
	{
		const auto& baseLayer = *layer.getBaseLayer();

		m_OptimizeScheduled_BaseLayer_tbl.push_back(layer.getBaseLayer());
		for (auto iter = baseLayer.getInternalLayerTbl().begin(), end = baseLayer.getInternalLayerTbl().end(); iter != end; iter++)
		{
			//layer::Layer layer{iter->second.getLayerCore(), ""};
			//(*this)(layer);
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

		optimize_unique();
	}

}
