#pragma once
#include "BaseLayer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class  AffineCore : public  BaseLayer
			{
			public:
				AffineCore(u32 output_size, DataType affineWeight = 0.1f);
				~AffineCore();

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;


				const DataType mAffineWeight = 0.1f;
				const u32 m_output_size;
				u32 m_batch_size;
				u32 m_input_size;
				TensorCore& mOutput;
				TensorCore& mWeight;
				TensorCore& mBias;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl_input(TensorCore&);
				void backward_cpu_impl_parameter(const TensorCore&);
			};


			Layer Affine(u32 output_size);
		}
	}
}

