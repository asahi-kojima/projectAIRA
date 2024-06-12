#pragma once
#include "Tensor/Tensor.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class Layer::nnLayer
			{

			public:
				template <typename T, typename ... Args>
				friend Layer::nnLayer gen(Args ... args);
				template <typename T, typename ... Args>
				friend Layer::nnLayer gen(const char* layerName, Args ... args);

				nnLayer() {}
				nnLayer(const nnLayer&);
				nnLayer(const std::shared_ptr<Layer::LayerSkeleton>&, std::string);

				Layer::LayerSkeleton::iotype operator()(const Layer::LayerSkeleton::iotype& input) const;

				template <typename ... Args>
				Layer::LayerSkeleton::iotype operator()(Args ... args) const
				{
					tensor::Tensor tensor_tbl[] = { args... };
					const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

					LayerSkeleton::iotype input_tensor_as_vector(input_tensor_num);

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

				const std::shared_ptr<Layer::LayerSkeleton>& getLayerCore() const
				{
					return mLayerSkeleton;
				}
				std::shared_ptr<Layer::LayerSkeleton> getLayerCore()
				{
					return mLayerSkeleton;
				}

			private:
				std::shared_ptr<Layer::LayerSkeleton> mLayerSkeleton;
				std::string mLayerName;
			};

			template <typename T, typename ... Args>
			Layer::nnLayer gen(Args ... args)
			{
				Layer::nnLayer layer{};
				layer.mLayerSkeleton = std::make_shared<T>(args...);
				return layer;
			}

			template <typename T, typename ... Args>
			Layer::nnLayer gen(const char* layerName, Args ... args)
			{
				Layer::nnLayer layer{};
				layer.mLayerSkeleton = std::make_shared<T>(args...);
				layer.mLayerName = layerName;
				return layer;
			}
		}
	}
}
