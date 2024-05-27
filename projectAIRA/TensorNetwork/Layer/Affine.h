#pragma once
#include "Layer.h"


class AffineCore : public LayerCore
{
public:
	AffineCore(u32 input_size, u32  output_size);
	~AffineCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override
	{
		std::cout << "Affine forward" << std::endl;

		auto t = input_tensors;
		//for (const auto& inner_layer : mInnerLayerCoreTbl)
		//{
		//	t = inner_layer->forward(t);
		//}

		return t;
	}
};


Layer Affine(u32 input_size, u32 output_size);