#pragma once
#include "Layer.h"


class AffineCore : public LayerCore
{
public:
	AffineCore(u32 input_size, u32  output_size) : LayerCore(1, 1) {}
	~AffineCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override
	{
		std::cout << "Affine forward" << std::endl;
		return iotype(1);
	}
};


Layer Affine(u32 input_size, u32 output_size)
{
	return gen<AffineCore>("Affine", input_size, output_size);
}