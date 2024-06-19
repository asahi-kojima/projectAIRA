#pragma once
#include "BaseLayer.h"
#include "Layer.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class AddCore : public BaseLayer
			{
			public:
				AddCore();
				~AddCore() {}

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				TensorCore& mOutput;
				u32 m_data_size;

				void forward_cpu_impl(const TensorCore&, const TensorCore&);
				void backward_cpu_impl( TensorCore&,  TensorCore&);
			};
			
			
			Layer Add();


			class AddAsInnerCore : public BaseLayer
			{
			public:
				AddAsInnerCore() : BaseLayer(2, 1)
				{
					mlayer["add"] = Add();
				}

				virtual iotype forward(const iotype& input_tensors) override
				{
					return mlayer["add"](input_tensors);
				}

			};



			inline Layer AddAsInner()
			{
				return gen<AddAsInnerCore>("AddAsInner");
			}
		}
	}
}



