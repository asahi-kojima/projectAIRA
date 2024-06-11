#pragma once
#include "Optimizer.h"

namespace aoba { namespace nn { namespace optimizer { class SGD; } } }

class aoba::nn::optimizer::SGD : public Optimizer
{
public:
	SGD(u32 learningRate);
	~SGD();

	virtual void optimize_unique() override;
};
