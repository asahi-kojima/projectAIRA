#include "Optimizer.h"

using OptimizerSkeleton = aoba::nn::optimizer::Optimizer::OptimizerSkeleton;


OptimizerSkeleton::OptimizerSkeleton(DataType learningRate)
	:mLearningRate(learningRate)
	, m_OptimizeScheduled_LayerCore_tbl(0)
	//, mLinkedParameters(nullptr)
	, mIsInitialized(false)
{
}

OptimizerSkeleton::~OptimizerSkeleton()
{
	//LinkedList* current = mLinkedParameters;
	//while (true)
	//{
	//	auto next = current->next;
	//	delete current;

	//	if (!next)
	//	{
	//		break;
	//	}
	//}
}

void OptimizerSkeleton::operator()(const layer::Layer::nnLayer& layer)
{
	const auto& layercore = *layer.getLayerCore();

	m_OptimizeScheduled_LayerCore_tbl.push_back(layer.getLayerCore());
	for (auto iter = layercore.m_internal_layer_tbl.begin(), end = layercore.m_internal_layer_tbl.end(); iter != end; iter++)
	{
		layer::Layer::nnLayer layer{iter->second, ""};
		(*this)(layer);
	}
}

void OptimizerSkeleton::optimize()
{
	//LinkedList* old = nullptr;
	//LinkedList* current = mLinkedParameters;

	if (!mIsInitialized)
	{
		initialize();
		mIsInitialized = true;
	}

	optimize_unique();
}
