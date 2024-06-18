#pragma once
#include "LayerBase.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class SplitCore : public LayerBase
			{
			public:
				SplitCore();
				~SplitCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors)override;
				virtual void backward() override;
			};


			Layer Split();
		}
	}
}
