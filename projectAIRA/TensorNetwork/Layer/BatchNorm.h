#pragma once
#include "BaseLayer.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class  BatchNormCore : public  BaseLayer
			{
			public:
				BatchNormCore();
				~BatchNormCore();

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;


				/*const DataType mAffineWeight = 0.1f;
				const u32 m_output_size;
				u32 m_batch_size;
				u32 m_input_size;*/

				TensorCore& mOutput;
				TensorCore& mGamma;
				TensorCore& mBeta;

				TensorCore mIntermediate;
				TensorCore mSigma;
				TensorCore mMean;

				TensorCore mBlockMean;
				TensorCore mBlockSqMean;


				u32 mBatchSize;
				u32 mIc;
				u32 mIhIw;
				u32 mIcIhIw;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);
			};


			Layer BatchNorm();
		}
	}
}

