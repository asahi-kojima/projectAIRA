#pragma once
#include "Layer.h"

namespace aoba { namespace nn { namespace layer { class AddCore; } } }

class aoba::nn::layer::AddCore : public LayerCore
{
public:
	AddCore();
	~AddCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;
};


Layer Add();
