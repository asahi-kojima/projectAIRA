#pragma once
#include "LayerBase.h"
#include "Layer.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class AddCore : public LayerBase
			{
			public:
				AddCore();
				~AddCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;
			};
			
			
			Layer Add();


			class AddAsInnerCore : public LayerBase
			{
			public:
				AddAsInnerCore() : LayerBase(2, 1)
				{
					mlayer["add"] = Add().getLayerCore();
				}

				virtual iotype forward(const iotype& input_tensors) override
				{
					return mlayer["add"]->callForward(input_tensors);
				}

			};



			inline Layer AddAsInner()
			{
				return gen<AddAsInnerCore>("AddAsInner");
			}
		}
	}
}



