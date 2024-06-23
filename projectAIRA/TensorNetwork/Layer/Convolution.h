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

				struct parameterInfo
				{
					u32 Fn;
					u32 Fh;
					u32 Fw;
					u32 FhFw;
					u32 Sh;
					u32 Sw;
					u32 Ph;
					u32 Pw;

					u32 batchSize;
					u32 Ic;
					u32 Ih;
					u32 Iw;

					u32 IcFhFw;
					u32 IhIw;
					u32 IcIhIw;

					u32 Oc;
					u32 Oh;
					u32 Ow;
					u32 OhOw;
					u32 OcOhOw;
					u32 OhOwIcFhFw;
				};
			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;


				//コンストラクタで決定できる変数
				const u32 mFn;
				const u32 mFh;
				const u32 mFw;
				const u32 mFhFw;
				const u32 mSh;
				const u32 mSw;
				const u32 mPh;
				const u32 mPw;
				const DataType mConvolutionWeight = 0.01f;

				TensorCore& mOutput;
				TensorCore& mWeight;
				TensorCore& mBias;

				//入力依存の変数
				TensorCore mReshapedInputData;

				u32 mBatchSize;
				u32 mIc;
				u32 mIh;
				u32 mIw;

				u32 mIcFhFw;
				u32 mIhIw;
				u32 mIcIhIw;

				u32 mOc;
				u32 mOh;
				u32 mOw;
				u32 mOhOw;
				u32 mOcOhOw;

				parameterInfo* mParameterInfoOnGPU;


				//const u32 m_output_size;
				//u32 m_batch_size;
				//u32 m_input_size;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl_input(TensorCore&);
				void backward_cpu_impl_parameter();

				void resetVariable(const TensorCore& input)
				{
					mBatchSize = input.getBatchSize();
					mIc = input.getChannel();
					mIh = input.getHeight();
					mIw = input.getWidth();

					mIcFhFw = mIc * mFh * mFw;
					mIhIw = mIh * mIw;
					mIcIhIw = mIc * mIhIw;

					mOc = mFn;//これはコンパイル時に決定できるので、ここではなくてもいい。
					mOh = 1 + (mIh - mFh + 2 * mPh) / mSh;
					mOw = 1 + (mIw - mFw + 2 * mPw) / mSw;
					mOhOw = mOh * mOw;
					mOcOhOw = mOc * mOhOw;


					if (input.isOnCuda())
					{
						//GPU用の変数を準備する。
						parameterInfo tmp;

						//コンパイル時に決定できる変数
						tmp.Fn = mFn;
						tmp.Fh = mFh;
						tmp.Fw = mFw;
						tmp.FhFw = mFhFw;
						tmp.Sh = mSh;
						tmp.Sw = mSw;
						tmp.Ph = mPh;
						tmp.Pw = mPw;

						//入力依存の変数
						tmp.batchSize = mBatchSize;
						tmp.Ic = mIc;
						tmp.Ih = mIh;
						tmp.Iw = mIw;

						tmp.IcFhFw = mIcFhFw;
						tmp.IhIw = mIhIw;
						tmp.IcIhIw = mIcIhIw;

						tmp.Oc = mOc;
						tmp.Oh = mOh;
						tmp.Ow = mOw;
						tmp.OhOw = mOhOw;
						tmp.OcOhOw = mOc * mOhOw;
						tmp.OhOwIcFhFw = mOhOw * mIcFhFw;

						CHECK(cudaMemcpy(mParameterInfoOnGPU, &tmp, sizeof(parameterInfo), cudaMemcpyHostToDevice));
					}
				}



				
			};


			Layer Convolution(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight = 0.01f);
			Layer Convolution(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight = 0.01f);
		}
	}
}