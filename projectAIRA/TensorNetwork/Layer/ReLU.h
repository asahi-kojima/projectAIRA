#pragma once
#include "Layer.h"
#include "Tensor/Tensor.h"
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

				//const u32 m_output_size;
				//u32 m_batch_size;
				//u32 m_input_size;
				u32 mDataSize;
				TensorCore& mOutput;
				TensorCore mMask;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);
			};


			Layer::nnLayer ReLU();
		}
	}
}




