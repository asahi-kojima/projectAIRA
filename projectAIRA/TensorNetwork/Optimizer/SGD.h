#pragma once
#include "BaseOptimizer.h"

class aoba::nn::optimizer::SGD : public aoba::nn::optimizer::BaseOptimizer
{

public:
	SGD(DataType learningRate);
	~SGD();

	virtual void initialize() override;
	virtual void optimize_unique() override;
};