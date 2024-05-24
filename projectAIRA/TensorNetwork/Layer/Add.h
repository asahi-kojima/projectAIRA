#pragma once
#include "Layer.h"


class AddCore : public LayerCore
{
public:
	AddCore(u32 input_size, u32  output_size) : LayerCore(2, 1) {}
	~AddCore() {}

private:
	virtual iotype forward(const iotype& input_tensors) override
	{
		std::cout << "Add forward" << std::endl;
		return iotype();
	}
};


Layer Add(u32 input_size, u32 output_size)
{
	return gen<AddCore>(input_size, output_size);
}