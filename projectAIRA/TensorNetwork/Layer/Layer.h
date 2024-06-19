#pragma once
#include "Tensor/Tensor.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class Layer
			{

			public:
				template <typename T, typename ... Args>
				friend Layer gen(Args ... args);
				template <typename T, typename ... Args>
				friend Layer gen(const char* layerName, Args ... args);

				Layer() {}
				Layer(const Layer&);
				Layer(const std::shared_ptr<BaseLayer>&, std::string);

				BaseLayer::iotype operator()(const BaseLayer::iotype& input) const;

				template <typename ... Args>
				BaseLayer::iotype operator()(Args ... args) const
				{
					tensor::Tensor tensor_tbl[] = { args... };
					const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

					BaseLayer::iotype input_tensor_as_vector(input_tensor_num);

					for (u32 i = 0, end = input_tensor_num; i < end; i++)
					{
						input_tensor_as_vector[i] = tensor_tbl[i];
					}

					return mBaseLayer->callForward(input_tensor_as_vector);
				}
				u32 get_input_tensor_num() const
				{
					return mBaseLayer->get_input_tensor_num();
				}

				u32 get_output_tensor_num() const
				{
					return mBaseLayer->get_output_tensor_num();
				}

				const std::shared_ptr<BaseLayer>& getBaseLayer() const
				{
					return mBaseLayer;
				}
				std::shared_ptr<BaseLayer> getBaseLayer()
				{
					return mBaseLayer;
				}

			private:
				std::shared_ptr<BaseLayer> mBaseLayer;
				std::string mLayerName;
			};

			template <typename T, typename ... Args>
			Layer gen(Args ... args)
			{
				Layer layer{};
				layer.mBaseLayer = std::make_shared<T>(args...);
				return layer;
			}

			template <typename T, typename ... Args>
			Layer gen(const char* layerName, Args ... args)
			{
				Layer layer{};
				layer.mBaseLayer = std::make_shared<T>(args...);
				layer.mLayerName = layerName;
				return layer;
			}
		}
	}
}
