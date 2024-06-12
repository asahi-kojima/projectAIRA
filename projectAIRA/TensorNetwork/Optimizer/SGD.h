#pragma once
#include "Optimizer.h"

//namespace aoba { namespace nn { namespace optimizer { class Optimizer::SGD; } } }

//class aoba::nn::optimizer::Optimizer::SGD : public Optimizer
//{
//public:
//	SGD(u32 learningRate);
//	~SGD();
//
//	virtual void initialize() override;
//	virtual void optimize_unique() override;
//};
class aoba::nn::optimizer::Optimizer::SGD : public aoba::nn::optimizer::Optimizer::OptimizerSkeleton
{

public:
	SGD(u32 learningRate);
	~SGD();

	virtual void initialize() override;
	virtual void optimize_unique() override;
};