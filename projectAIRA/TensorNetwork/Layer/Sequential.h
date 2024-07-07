#pragma once
#include <string>
#include "BaseLayer.h"
#include "Layer.h"

namespace aoba
{
	namespace nn 
	{
		namespace layer
		{
			class SequentialCore : public BaseLayer
			{
			public:
				template <typename ... Args>
				SequentialCore(Args ... args) : BaseLayer(1, 1)
				{
					//可変長テンプレートなので、分解してinnerModuleに格納している。
					Layer layer_tbl[] = { args... };
					const u32 inner_layer_num = (sizeof(layer_tbl) / sizeof(layer_tbl[0]));

					for (u32 i = 0, end = inner_layer_num; i < end; i++)
					{
						if (layer_tbl[i].get_input_tensor_num() != 1 || layer_tbl[i].get_output_tensor_num() != 1)
						{
							std::cout << "inner module of Sequential  must be 1 input(your : "
								<< layer_tbl[i].get_input_tensor_num() << " ) and 1 output(your : "
								<< layer_tbl[i].get_output_tensor_num() << " ). " << std::endl;
							exit(1);
						}
						//const std::string layerId = layer_tbl[i].getLayerName() + "#" + std::to_string(i);
						mlayer[std::to_string(i)] = layer_tbl[i];
					}
				}

				virtual iotype forward(const iotype& input) override
				{
					if (input.size() != 1)
					{
						std::cout << "input tensor num is not 1" << std::endl;
					}
					iotype tensor = input;
					for (u32 i = 0, end = mlayer.size(); i < end; i++)
					{
						tensor = mlayer[std::to_string(i)](tensor);
					}
					return tensor;
				}

			};

			template<typename ... Args>
			Layer Sequential(Args ... args)
			{
				return gen<SequentialCore>("Sequential", args...);
			}

		}
	}
}



