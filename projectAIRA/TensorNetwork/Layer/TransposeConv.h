#pragma once
#include "BaseLayer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class  TransposeConvCore : public  BaseLayer
			{
			public:
				TransposeConvCore(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 weight = 0.01f);
				TransposeConvCore(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 weight = 0.01f);
				~TransposeConvCore();

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
				const DataType mTransposeConvWeight = 0.01f;

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
				bool mIsParamerInfoAllocated = false;


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
					mOh = (mIh - 1) * mSh + mFh - 2 * mPh;
					mOw = (mIw - 1) * mSw + mFw - 2 * mPw;
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
						if (!mIsParamerInfoAllocated)
						{
							CHECK(cudaMalloc(&mParameterInfoOnGPU, sizeof(parameterInfo)));
							mIsParamerInfoAllocated = true;
						}
						CHECK(cudaMemcpy(mParameterInfoOnGPU, &tmp, sizeof(parameterInfo), cudaMemcpyHostToDevice));
					}
				}




			};


			Layer TransposeConv(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight = 0.01f);
			Layer TransposeConv(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight = 0.01f);
		}
	}
}