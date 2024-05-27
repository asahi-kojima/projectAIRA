#pragma once
#include "Layer.h"


class SequentialCore : public LayerCore
{
public:
	template <typename ... Args>
	SequentialCore(Args ... args) : LayerCore(1, 1)
	{
		//可変長テンプレートなので、分解してinnerModuleに格納している。
		Layer layer_tbl[] = { args... };
		const u32 inner_layer_num = (sizeof(layer_tbl) / sizeof(layer_tbl[0]));
		mInnerLayer.resize(inner_layer_num);

		for (u32 i = 0, end = inner_layer_num; i < end; i++)
		{
			if (layer_tbl[i].mLayerCore->get_input_tensor_num() != 1 || layer_tbl[i].mLayerCore->get_output_tensor_num() != 1)
			{
				std::cout << "inner module of Sequential  must be 1 input(your : "
					<< layer_tbl[i].mLayerCore->get_input_tensor_num() << " ) and 1 output(your : "
					<< layer_tbl[i].mLayerCore->get_output_tensor_num() << " ). " << std::endl;
				exit(1);
			}
			mInnerLayer[i] = layer_tbl[i];
		}
	}

	virtual iotype forward(const iotype& input) override
	{
		if (input.size() != 1)
		{
			std::cout << "input tensor num is not 1" << std::endl;
		}
		iotype tensor = input;
		for (const auto& layer : mInnerLayer)
		{
			tensor = layer(tensor);
		}
		return tensor;
	}

private:
	std::vector<Layer> mInnerLayer;
};

template<typename ... Args>
Layer Sequential(Args ... args)
{
	return gen<SequentialCore>("Sequential", args...);
}