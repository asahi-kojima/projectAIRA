#pragma once
#include "Layer.h"

namespace aoba 
{
	namespace nn
	{
		namespace layer 
		{
			class Layer::CrossEntropyWithSMCore : public Layer::LayerSkeleton
			{
			public:
				CrossEntropyWithSMCore();
				~CrossEntropyWithSMCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				u32 m_batch_size;
				u32 m_label_num;
				void crossEntropyWithSM_forward_cpu_impl(const Layer::LayerSkeleton::iotype&);
			};


			Layer::nnLayer CrossEntropyWithSM();


		}
	}
}

