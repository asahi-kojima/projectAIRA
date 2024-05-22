#pragma once
#include "Layer.h"


class nnSplit : public LayerCore
{
public:
	nnSplit() : LayerCore(1, 2) {}
	~nnSplit() {}

private:
	virtual iotype forward(const iotype& input_tensors)override
	{
		return iotype{ Tensor{}, Tensor{} };
	}
};


Layer Split()
{
	return gen<nnSplit>();
}
