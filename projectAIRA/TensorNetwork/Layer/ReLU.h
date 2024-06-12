#pragma once
#include "Layer.h"

namespace aoba 
{
	namespace nn
	{
		namespace layer
		{

			class Layer::ReLUCore : public Layer::LayerSkeleton
			{
			public:
				ReLUCore();
				~ReLUCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;
			};


			Layer::nnLayer ReLU();
		}
	}
}




