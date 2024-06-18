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
				Layer(const std::shared_ptr<LayerBase>&, std::string);

				LayerBase::iotype operator()(const LayerBase::iotype& input) const;

				template <typename ... Args>
				LayerBase::iotype operator()(Args ... args) const
				{
					tensor::Tensor tensor_tbl[] = { args... };
					const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

					LayerBase::iotype input_tensor_as_vector(input_tensor_num);

					for (u32 i = 0, end = input_tensor_num; i < end; i++)
					{
						input_tensor_as_vector[i] = tensor_tbl[i];
					}

					return mLayerSkeleton->callForward(input_tensor_as_vector);
				}
				u32 get_input_tensor_num() const
				{
					return mLayerSkeleton->get_input_tensor_num();
				}

				u32 get_output_tensor_num() const
				{
					return mLayerSkeleton->get_output_tensor_num();
				}

				const std::shared_ptr<LayerBase>& getLayerCore() const
				{
					return mLayerSkeleton;
				}
				std::shared_ptr<LayerBase> getLayerCore()
				{
					return mLayerSkeleton;
				}

			private:
				std::shared_ptr<LayerBase> mLayerSkeleton;
				std::string mLayerName;
			};

			template <typename T, typename ... Args>
			Layer gen(Args ... args)
			{
				Layer layer{};
				layer.mLayerSkeleton = std::make_shared<T>(args...);
				return layer;
			}

			template <typename T, typename ... Args>
			Layer gen(const char* layerName, Args ... args)
			{
				Layer layer{};
				layer.mLayerSkeleton = std::make_shared<T>(args...);
				layer.mLayerName = layerName;
				return layer;
			}
		}
	}
}
