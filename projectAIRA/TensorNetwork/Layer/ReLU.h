#pragma once
#include "BaseLayer.h"
#include "Tensor/Tensor.h"
namespace aoba 
{
	namespace nn
	{
		namespace layer
		{

			class ReLUCore : public BaseLayer
			{
			public:
				ReLUCore();
				~ReLUCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				u32 mDataSize;
				TensorCore& mOutput;
				TensorCore mMask;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);
			};


			Layer ReLU();
		}
	}
}




