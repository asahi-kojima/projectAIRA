#pragma once
#include "Layer.h"


class AffineCore : public LayerCore
{
public:
	AffineCore(u32 input_size, u32  output_size);
	~AffineCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;
};


Layer Affine(u32 input_size, u32 output_size);