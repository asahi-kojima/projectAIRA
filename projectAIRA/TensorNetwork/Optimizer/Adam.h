#pragma once
#include "BaseOptimizer.h"

namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			class Adam : public BaseOptimizer
			{
			public:
				Adam(DataType learningRate, DataType beta0 = 0.9f, DataType beta1=0.9f);
				~Adam();

				virtual void initialize() override;
				virtual void optimize_unique(tensor::TensorCore&) override;

			private:
				//std::vector<tensor::TensorCore> mMomentumTbl;
				//std::vector<tensor::TensorCore> mVelocityTbl;
				std::map<tensor::TensorCore*, tensor::TensorCore> mMomentumMap;
				std::map<tensor::TensorCore*, tensor::TensorCore> mVelocityMap;
				//std::map<tensor::TensorCore*, u32> mOrderMap;
				DataType mBeta0;
				DataType mBeta1;
				std::map<tensor::TensorCore* ,u32> mIterationTbl;
			};
		}
	}
}