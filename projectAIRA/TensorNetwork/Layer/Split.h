#pragma once
#include "Layer.h"


class SplitCore : public LayerCore
{
public:
	SplitCore();
	~SplitCore() {}

private:
	virtual iotype forward(const iotype& input_tensors)override;
	virtual void backward() override;
};


Layer Split();
