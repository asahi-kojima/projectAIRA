#pragma once
#include "BaseLayer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class SplitCore : public BaseLayer
			{
			public:
				SplitCore();
				~SplitCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors)override;
				virtual void backward() override;

				u32 m_data_size;
				TensorCore& mOutput0;
				TensorCore& mOutput1;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);
			};


			Layer Split();
		}
	}
}
