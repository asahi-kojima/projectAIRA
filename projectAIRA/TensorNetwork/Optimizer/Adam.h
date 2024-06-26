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
				Adam(DataType learningRate);
				~Adam();

				virtual void initialize() override;
				virtual void optimize_unique() override;

			private:
				DataType mBeta0;
				DataType mBeta1;
				u32 mIteration;
			};
		}
	}
}