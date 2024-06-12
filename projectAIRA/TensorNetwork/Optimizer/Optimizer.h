#pragma once
#include <memory>
#include "Layer/Layer.h"
#include "Layer/nnLayer.h"
#include "Tensor/Tensor.h"
namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			class Optimizer;
			class OptimizerSkeleton;
		}
	}
}


class aoba::nn::optimizer::Optimizer
{
public:
	class OptimizerSkeleton;
	class SGD;
	class Adam;
};

class aoba::nn::optimizer::Optimizer::OptimizerSkeleton
{
public:
	OptimizerSkeleton(u32 learningRate);
	virtual ~OptimizerSkeleton();

	virtual void optimize() final;
	virtual void initialize() = 0;
	virtual void optimize_unique() = 0;
	virtual void operator()(const layer::Layer::nnLayer&) final;

	class SGD;
	class Adam;

protected:
	//using TensorCore = nn::tensor::TensorCore;
	//class LinkedList
	//{
	//public:
	//	LinkedList(const std::shared_ptr<TensorCore>& tensorcore) :parameter(tensorcore), next(nullptr) {}
	//	std::weak_ptr<TensorCore> parameter;
	//	LinkedList* next;
	//};

	DataType mLearningRate;
	std::vector<std::weak_ptr<layer::Layer::LayerSkeleton> > m_OptimizeScheduled_LayerCore_tbl;
	//LinkedList* mLinkedParameters;

private:
	bool mIsInitialized;
};


