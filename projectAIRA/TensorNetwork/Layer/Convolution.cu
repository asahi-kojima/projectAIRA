#include <random>
#include "Layer.h"
#include "Convolution.h"


namespace
{

}



namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer Convolution(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight)
			{
				Layer conv = gen<ConvolutionCore>("Convolution", filterNum, filterSize, stride, padding, convWeight);
				return conv;
			}

			Layer Convolution(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight)
			{
				Layer conv = gen<ConvolutionCore>("Convolution", filterNum, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, convWeight);
				return conv;
			}



			ConvolutionCore::ConvolutionCore(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight)
				:ConvolutionCore(filterNum, filterSize, filterSize, stride, stride, padding, padding, convWeight)
			{

			}

			ConvolutionCore::ConvolutionCore(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight)
				:BaseLayer(1, 1, 1, 2)

				, mFn(filterNum)
				, mFh(filterHeight)
				, mFw(filterWidth)
				, mSh(strideHeight)
				, mSw(strideWidth)
				, mPh(paddingHeight)
				, mPw(paddingWidth)
				, mConvolutionWeight(convWeight)

				, mOutput(*m_output_tensorcore_tbl[0])
				, mWeight(*mTrainableParameterTbl[0])
				, mBias(*mTrainableParameterTbl[1])
			{
			}

			ConvolutionCore::~ConvolutionCore()
			{
			}


			BaseLayer::iotype ConvolutionCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}

				const auto& input = *getTensorCoreFrom(input_tensors[0]);

				{
					//mOutput.reshapeAs();
				}


				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}

			void ConvolutionCore::backward()
			{

			}
		}
	}
}