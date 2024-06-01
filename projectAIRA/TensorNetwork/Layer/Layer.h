#pragma once
#include "LayerCore.h"


class Layer
{
public:
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


	//void save() {mLayerCore}
	//void load();

	std::shared_ptr<LayerCore> mLayerCore;
	std::string mLayerName;
private:
};

template <typename T, typename ... Args>
Layer gen(Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	return layer;
}

template <typename T, typename ... Args>
Layer gen(const char* layerName, Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	layer.mLayerName = layerName;
	return layer;
}


