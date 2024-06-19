#pragma once
#include "BaseLayer.h"
#include "Tensor/Tensor.h"
namespace aoba 
{
	namespace nn
	{
		namespace layer 
		{
			class CrossEntropyWithSMCore : public BaseLayer
			{
			public:
				CrossEntropyWithSMCore();
				~CrossEntropyWithSMCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				u32 m_batch_size;
				u32 m_label_num;
				TensorCore& mOutput;
				TensorCore mLossPerBatch;


				void forward_cpu_impl(const TensorCore&, const TensorCore&);
				void backward_cpu_impl(TensorCore&, const TensorCore&);
			};


			Layer CrossEntropyWithSM();


		}
	}
}

