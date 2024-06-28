#pragma once
#include <functional>
#include "BaseLayer.h"
#include "Layer/Layer.h"

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer Tanh();
			Layer Sigmoid();



			class BasisFunctionCore : public BaseLayer
			{
			public:
				using CPUFunctionF = std::function<DataType(DataType)>;
				using GPUFunctionF = void(*)(DataType*, const DataType* ,u32 dataSize);

				using CPUFunctionB = std::function<DataType(DataType, DataType)>;
				using GPUFunctionB = void(*)(DataType*, const DataType* , const DataType*, u32 dataSize);


				BasisFunctionCore(CPUFunctionF, GPUFunctionF, CPUFunctionB, GPUFunctionB);
				~BasisFunctionCore();

			private:
				virtual iotype forward(const iotype& input_tensors) override;
				virtual void backward() override;

				u32 mDataSize;
				TensorCore& mOutput;

				CPUFunctionF mFunctionCPU_forward;
				GPUFunctionF mFunctionGPU_forward;
				CPUFunctionB mFunctionCPU_backward;
				GPUFunctionB mFunctionGPU_backward;

				void forward_cpu_impl(const TensorCore&);
				void backward_cpu_impl(TensorCore&);
			};
		}
	}
}




