#pragma once
#include "Layer.h"


class SplitCore : public LayerCore
{
public:
	SplitCore() : LayerCore(1, 2) {}
	~SplitCore() {}

private:
	virtual iotype forward(const iotype& input_tensors)override
	{
		return iotype{ Tensor{}, Tensor{} };
	}
};


Layer Split()
{
	return gen<SplitCore>();
}
