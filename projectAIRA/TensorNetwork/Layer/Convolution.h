#pragma once
#include "BaseLayer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class  ConvolutionCore : public  BaseLayer
			{
			public:
				ConvolutionCore(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 weight = 0.01f);
				ConvolutionCore(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 weight = 0.01f);
				~ConvolutionCore();

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;


				//コンストラクタで決定できる変数
				const u32 mFn;
				const u32 mFh;
				const u32 mFw;
				const u32 mSh;
				const u32 mSw;
				const u32 mPh;
				const u32 mPw;
				const DataType mConvolutionWeight = 0.1f;

				TensorCore& mOutput;
				TensorCore& mWeight;
				TensorCore& mBias;

				//入力依存の変数
				TensorCore mReshapedInputData;


				//const u32 m_output_size;
				//u32 m_batch_size;
				//u32 m_input_size;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl_input(TensorCore&);
				void backward_cpu_impl_parameter(const TensorCore&);
			};


			Layer Convolution(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight = 0.01f);
			Layer Convolution(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight = 0.01f);
		}
	}
}