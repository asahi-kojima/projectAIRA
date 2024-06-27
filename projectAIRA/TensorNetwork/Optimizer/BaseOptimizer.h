#pragma once
#include <memory>
#include "Layer/BaseLayer.h"
#include "Layer/Layer.h"
#include "Tensor/Tensor.h"
namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			class BaseOptimizer;
			class SGD;
			class Adam;
		}

		namespace tensor
		{
			class TensorCore;
		}
	}
}


namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			class BaseOptimizer
			{
			public:
				BaseOptimizer(DataType learningRate);
				virtual ~BaseOptimizer();

				virtual void optimize() final;
				virtual void initialize() = 0;
				virtual void optimize_unique(tensor::TensorCore&) = 0;
				virtual void operator()(const layer::Layer&) final;

				inline static DataType convert_loss_to_prob(DataType loss) { return exp(-loss); }
			protected:

				DataType mLearningRate;
				//std::vector<std::weak_ptr<layer::BaseLayer> > m_OptimizeScheduled_BaseLayer_tbl;
				std::vector<std::weak_ptr<tensor::TensorCore> > mOptimizeScheduledParamTbl;

			private:
				bool mIsInitialized;
			};

		}
	}
}


