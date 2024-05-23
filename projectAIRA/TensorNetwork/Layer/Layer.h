#pragma once
#include "LayerCore.h"


class Layer
{
public:
	std::shared_ptr<LayerCore> mLayerCore;
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

		return mLayerCore->forward(input_tensor_as_vector);
	}

	LayerCore::iotype operator()(const LayerCore::iotype& input)
	{
		return mLayerCore->forward(input);
	}
};

template <typename T, typename ... Args>
Layer gen(Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	return layer;
}