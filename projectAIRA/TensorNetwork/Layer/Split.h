#pragma once
#include "Layer.h"

namespace aoba { namespace nn { namespace layer { class SplitCore; } } }

class aoba::nn::layer::SplitCore : public LayerCore
{
public:
	SplitCore();
	~SplitCore() {}

private:
	virtual iotype forward(const iotype& input_tensors)override;
	virtual void backward() override;
};


Layer Split();
