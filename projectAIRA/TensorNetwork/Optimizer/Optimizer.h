#pragma once
#include <memory>
#include "Layer/LayerBase.h"
#include "Layer/Layer.h"
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

	inline static DataType convert_loss_to_prob(DataType loss) { return exp(-loss); }
};

class aoba::nn::optimizer::Optimizer::OptimizerSkeleton
{
public:
	OptimizerSkeleton(DataType learningRate);
	virtual ~OptimizerSkeleton();

	virtual void optimize() final;
	virtual void initialize() = 0;
	virtual void optimize_unique() = 0;
	virtual void operator()(const layer::Layer&) final;

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
	std::vector<std::weak_ptr<layer::LayerBase> > m_OptimizeScheduled_LayerCore_tbl;
	//LinkedList* mLinkedParameters;

private:
	bool mIsInitialized;
};


