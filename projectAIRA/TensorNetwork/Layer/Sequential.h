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
		const u32 inner_module_num = (sizeof(layer_tbl) / sizeof(layer_tbl[0]));
		mInnerLayerCoreTbl.resize(inner_module_num);

		for (u32 i = 0, end = inner_module_num; i < end; i++)
		{
			if (layer_tbl[i].mLayerCore->get_input_tensor_num() != 1 || layer_tbl[i].mLayerCore->get_output_tensor_num() != 1)
			{
				std::cout << "inner module of Sequential  must be 1 input(your : "
					<< layer_tbl[i].mLayerCore->get_input_tensor_num() << " ) and 1 output(your : "
					<< layer_tbl[i].mLayerCore->get_output_tensor_num() << " ). " << std::endl;
				exit(1);
			}
			mInnerLayerCoreTbl[i] = layer_tbl[i].mLayerCore;
		}
	}

	virtual iotype forward(const iotype& input) override
	{
		if (input.size() != 1)
		{
			std::cout << "input tensor num is not 1" << std::endl;
		}

		return iotype(1);
	}
};

template<typename ... Args>
Layer Sequential(Args ... args)
{
	return gen<SequentialCore>("Sequential", args...);
}