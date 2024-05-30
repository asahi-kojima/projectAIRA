#pragma once
#include "Layer.h"

class ReLUCore : public LayerCore
{
public:
	ReLUCore();
	~ReLUCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;
};


Layer ReLU(); 