#pragma once
#include "BaseLayer.h"
#include "Tensor/Tensor.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class L2LossCore : public BaseLayer
			{
			public:
				L2LossCore();
				~L2LossCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				u32 mBatchSize;
				u32 mCHW;
				TensorCore& mOutput;
				TensorCore mLossPerBatch;

				void forward_cpu_impl(const TensorCore&, const TensorCore&);
				void backward_cpu_impl(TensorCore&, TensorCore&);
			};


			Layer L2Loss();

		}
	}
}