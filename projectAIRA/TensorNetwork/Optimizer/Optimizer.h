#pragma once
#include <memory>
#include "Layer/Layer.h"
#include "Tensor/Tensor.h"

namespace aoba { namespace nn { namespace optimizer { class Optimizer; } } }



class aoba::nn::optimizer::Optimizer
{
public:
	Optimizer(u32 learningRate);
	virtual ~Optimizer();

	void optimize();

	virtual void optimize_unique() = 0;
	virtual void operator()(const Layer&) final;

protected:
	using TensorCore = nn::tensor::TensorCore;
	class LinkedList
	{
	public:
		LinkedList(const std::shared_ptr<TensorCore>& tensorcore, LinkedList* next) :parameter(tensorcore), next(nullptr) {}
		std::weak_ptr<TensorCore> parameter;
		LinkedList* next;
	};

	DataType mLearningRate;
	LinkedList* mLinkedParameters;
};

