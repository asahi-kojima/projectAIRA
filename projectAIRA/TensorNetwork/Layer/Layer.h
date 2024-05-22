#pragma once
#include "LayerCore.h"


class Layer
{
public:
	std::shared_ptr<LayerCore> mModule;
	template <typename ... Args>
	LayerCore::iotype operator()(Args ... args)
	{
		Tensor tensor_tbl[] = { args... };
		const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

		LayerCore::iotype input_tensor_as_vector(input_tensor_num);

		for (u32 i = 0, end = input_tensor_num; i < end; i++)
		{
			input_tensor_as_vector[i] = tensor_tbl[i];
		}

		return mModule->forward(input_tensor_as_vector);
	}

	LayerCore::iotype operator()(const LayerCore::iotype& input)
	{
		return mModule->forward(input);
	}
};

template <typename T, typename ... Args>
Layer gen(Args ... args)
{
	Layer module{};
	module.mModule = std::make_shared<T>(args...);
	return module;
}