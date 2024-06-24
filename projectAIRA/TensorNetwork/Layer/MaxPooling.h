#pragma once
#include "BaseLayer.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			class MaxPoolingCore : public BaseLayer
			{
			public:
				MaxPoolingCore(u32 filterSize, u32 stride, u32 padding);
				MaxPoolingCore(
					u32 filterHeight, u32 filterWidth, 
					u32 strideHeight, u32 strideWidth, 
					u32 paddingHeight, u32 paddingWidth);
				~MaxPoolingCore();

				struct parameterInfo
				{
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
					//u32 OhOwIcFhFw;
				};

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				//�R���X�g���N�^�Ō���ł���ϐ�
				const u32 mFh;
				const u32 mFw;
				const u32 mFhFw;
				const u32 mSh;
				const u32 mSw;
				const u32 mPh;
				const u32 mPw;

				
				TensorCore& mOutput;

				//���͈ˑ��̕ϐ�
				//TensorCore mReshapedInputData;
				TensorCore mMaxLocation;

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



				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);

				void resetVariable(const TensorCore& input)
				{
					mBatchSize = input.getBatchSize();
					mIc = input.getChannel();
					mIh = input.getHeight();
					mIw = input.getWidth();

					mIcFhFw = mIc * mFh * mFw;
					mIhIw = mIh * mIw;
					mIcIhIw = mIc * mIhIw;

					mOc = mIc;
					mOh = 1 + (mIh - mFh + 2 * mPh) / mSh;
					mOw = 1 + (mIw - mFw + 2 * mPw) / mSw;
					mOhOw = mOh * mOw;
					mOcOhOw = mOc * mOhOw;


					if (input.isOnCuda())
					{
						//GPU�p�̕ϐ�����������B
						parameterInfo tmp;

						//�R���p�C�����Ɍ���ł���ϐ�
						tmp.Fh = mFh;
						tmp.Fw = mFw;
						tmp.FhFw = mFhFw;
						tmp.Sh = mSh;
						tmp.Sw = mSw;
						tmp.Ph = mPh;
						tmp.Pw = mPw;

						//���͈ˑ��̕ϐ�
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
						//tmp.OhOwIcFhFw = mOhOw * mIcFhFw;

						CHECK(cudaMemcpy(mParameterInfoOnGPU, &tmp, sizeof(parameterInfo), cudaMemcpyHostToDevice));
					}
				}
			};


			Layer MaxPooling(u32 filterSize, u32 stride, u32 padding);
			Layer MaxPooling(u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth);
		}
	}
}