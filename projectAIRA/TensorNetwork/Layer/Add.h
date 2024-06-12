#pragma once
#include "Layer.h"
#include "nnLayer.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class Layer::AddCore : public LayerSkeleton
			{
			public:
				AddCore();
				~AddCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;
			};
			
			
			Layer::nnLayer Add();


			class Layer::AddAsInnerCore : public Layer::LayerSkeleton
			{
			public:
				AddAsInnerCore() : LayerSkeleton(2, 1)
				{
					mlayer["add"] = Add().getLayerCore();
				}

				virtual iotype forward(const iotype& input_tensors) override
				{
					return mlayer["add"]->callForward(input_tensors);
				}

			};



			inline Layer::nnLayer AddAsInner()
			{
				return gen<Layer::AddAsInnerCore>("AddAsInner");
			}
		}
	}
}



