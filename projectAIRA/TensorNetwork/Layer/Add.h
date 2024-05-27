#pragma once
#include "Layer.h"


class AddCore : public LayerCore
{
public:
	AddCore();
	~AddCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;
};


Layer Add();