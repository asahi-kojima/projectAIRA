#pragma once
#include "Layer.h"

namespace aoba { namespace nn { namespace layer { class ReLUCore; } } }


class aoba::nn::layer::ReLUCore : public LayerCore
{
public:
	ReLUCore();
	~ReLUCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;
};


Layer ReLU();

