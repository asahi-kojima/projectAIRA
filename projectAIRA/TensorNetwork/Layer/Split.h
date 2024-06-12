#pragma once
#include "Layer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class Layer::SplitCore : public Layer::LayerSkeleton
			{
			public:
				SplitCore();
				~SplitCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors)override;
				virtual void backward() override;
			};


			Layer::nnLayer Split();
		}
	}
}
