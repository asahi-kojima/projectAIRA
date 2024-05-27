#pragma once
#include "LayerCore.h"


class Layer
{
public:
	std::shared_ptr<LayerCore> mLayerCore;

	LayerCore::iotype operator()(const LayerCore::iotype& input) const
	{
		return mLayerCore->callForward(input);
	}


	template <typename ... Args>
	LayerCore::iotype operator()(Args ... args) const
	{
		Tensor tensor_tbl[] = { args... };
		const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

		LayerCore::iotype input_tensor_as_vector(input_tensor_num);

		for (u32 i = 0, end = input_tensor_num; i < end; i++)
		{
			input_tensor_as_vector[i] = tensor_tbl[i];
		}

		return mLayerCore->callForward(input_tensor_as_vector);
	}


	std::string mLayerName;
};

template <typename T, typename ... Args>
Layer gen(Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	//layer.mLayerCore->regist_this_to_output_tensor();
	return layer;
}

template <typename T, typename ... Args>
Layer gen(const char* layerName, Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	//layer.mLayerCore->regist_this_to_output_tensor();
	layer.mLayerName = layerName;
	return layer;
}