#include <device_functions.h>
#include "Layer.h"
#include "MaxPooling.h"
#if !defined(__CUDACC__)
#define __CUDACC__
#endif

namespace
{

	__global__ void forward_gpu_impl(
		DataType* output,
		const DataType* input,
		DataType* mMaxLocation,
		aoba::nn::layer::MaxPoolingCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::MaxPoolingCore::parameterInfo& info = *pInfo;

		const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;

		if (N >= mBatchSize || OcOhOw >= mOcOhOw)
		{
			return;
		}

		//parameterInfoで決まるもの
		const u32 mFh = info.Fh;
		const u32 mFw = info.Fw;
		const u32 mSh = info.Sh;
		const u32 mSw = info.Sw;
		const u32 mPh = info.Ph;
		const u32 mPw = info.Pw;

		const u32 mIh = info.Ih;
		const u32 mIw = info.Iw;
		const u32 mIhIw = info.IhIw;
		const u32 mIcIhIw = info.IcIhIw;

		const u32 mOw = info.Ow;
		const u32 mOhOw = info.OhOw;


		//計算で決まるもの
		const u32 Oc = OcOhOw / mOhOw;
		const u32 Ic = Oc;
		const u32 OhOw = OcOhOw - Oc * mOhOw;
		const u32 Oh = OhOw / mOw;
		const u32 Ow = OhOw - Oh * mOw;

		//実際の計算
		const s32 basisIndexIh = Oh * mSh - mPh;
		const s32 basisIndexIw = Ow * mSw - mPw;

		bool paddingIncluded = false;

		const s32 indexHStartPoint = (basisIndexIh < 0) ? (paddingIncluded = true, 0) : basisIndexIh;
		const s32 indexWStartPoint = (basisIndexIw < 0) ? (paddingIncluded = true, 0) : basisIndexIw;

		const s32 indexHEndPoint = (indexHStartPoint + mFh >= mIh) ? (paddingIncluded = true, mIh) : indexHStartPoint + mFh;
		const s32 indexWEndPoint = (indexWStartPoint + mFw >= mIw) ? (paddingIncluded = true, mIw) : indexWStartPoint + mFw;

		s32 maxIh = indexHStartPoint;
		s32 maxIw = indexWStartPoint;
		DataType maxValue = input[N * mIcIhIw + Ic * mIhIw + maxIh * mIw + maxIw];

		for (u32 indexH = indexHStartPoint; indexH < indexHEndPoint; indexH++)
		{
			for (u32 indexW = indexWStartPoint; indexW < indexWEndPoint; indexW++)
			{
				DataType maxCandValue = input[N * mIcIhIw + Ic * mIhIw + indexH * mIw + indexW];
				if (maxValue < maxCandValue)
				{
					maxValue = maxCandValue;
					maxIh = indexH;
					maxIw = indexW;
				}
			}
		}

		s32 maxValueIndex = Ic * mIhIw + maxIh * mIw + maxIw;

		if (paddingIncluded)
		{
			if (maxValue < 0.0f)
			{
				maxValue = 0.0f;
				maxValueIndex = -1;
			}
		}

		mMaxLocation[N * mOcOhOw + OcOhOw] = maxValueIndex;
		output[N * mOcOhOw + OcOhOw] = maxValue;
	}

	__global__ void backward_gpu_impl_init(
		DataType* input_grad, const u32 mBatchSize, const u32 mIcIhIw)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;

		if (N >= mBatchSize || IcIhIw >= mIcIhIw)
		{
			return;
		}

		input_grad[N * mIcIhIw + IcIhIw] = 0.0f;
	}

	__global__ void backward_gpu_impl(
		DataType* input_grad,
		const DataType* output_grad,
		const DataType* mMaxLocation,
		aoba::nn::layer::MaxPoolingCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::MaxPoolingCore::parameterInfo& info = *pInfo;

		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;

		if (N >= mBatchSize || OcOhOw >= mOcOhOw)
		{
			return;
		}

		const u32 mIcIhIw = info.IcIhIw;


		const s32 location = static_cast<s32>(mMaxLocation[N * mOcOhOw + OcOhOw]);

		//locationが0以下ならパディング領域が最大だったことを意味するため、スキップできる。
		if (location < 0)
		{
			return;
		}

		//input_grad[N * mIcIhIw + location] += output_grad[N * mOcOhOw + OcOhOw];
		atomicAdd(&(input_grad[N * mIcIhIw + location]), output_grad[N * mOcOhOw + OcOhOw]);

	}
}


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer MaxPooling(u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth)
			{
				Layer maxPooling = gen<MaxPoolingCore>("MaxPooling", filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth);
				return maxPooling;
			}

			Layer MaxPooling(u32 filterSize, u32 stride, u32 padding)
			{
				Layer maxPooling = gen<MaxPoolingCore>("MaxPooling", filterSize, stride, padding);
				return maxPooling;
			}


			MaxPoolingCore::MaxPoolingCore(u32 filterSize, u32 stride, u32 padding)
				:MaxPoolingCore(filterSize, filterSize, stride, stride, padding, padding)
			{

			}

			MaxPoolingCore::MaxPoolingCore(u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth)
				:BaseLayer(1, 1, 1)
				//入力非依存
				, mFh(filterHeight)
				, mFw(filterWidth)
				, mFhFw(filterHeight* filterWidth)
				, mSh(strideHeight)
				, mSw(strideWidth)
				, mPh(paddingHeight)
				, mPw(paddingWidth)
				//入力依存
				, mOutput(*m_output_tensorcore_tbl[0])
				//ヘルパー
				, mMaxLocation(false)
			{
				CHECK(cudaMalloc(&mParameterInfoOnGPU, sizeof(parameterInfo)));
			}

			MaxPoolingCore::~MaxPoolingCore()
			{
				CHECK(cudaFree(mParameterInfoOnGPU));
			}


			BaseLayer::iotype MaxPoolingCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}

				const auto& input = *getTensorCoreFrom(input_tensors[0]);
				const bool input_on_cuda = input.isOnCuda();

				//入力の形状チェック
				{
					//入力が4次元でないと機能しない。これは仕様。
					if (input.getDimension() != TensorCore::Dimension::dim4)
					{
						assert(0);
					}
				}

				{
					//入力を参考に、変数の再設定を行う。
					resetVariable(input);

					mOutput.reshapeAs(mBatchSize, mOc, mOh, mOw, m_on_cuda);

					mMaxLocation.reshapeAs(mBatchSize, mOcOhOw, m_on_cuda);
				}

				{
					if (m_on_cuda)
					{
						auto output_gpu_address = mOutput.getGpuDataAddress();
						const auto input_gpu_address = input.getGpuDataAddress();
						auto maxLocation_gpu_address = mMaxLocation.getGpuDataAddress();

						dim3 block(32, 32);
						dim3 grid(
							(mBatchSize + block.x - 1) / block.x,
							(mOcOhOw + block.y - 1) / block.y);

						forward_gpu_impl << <grid, block >> > (
							output_gpu_address,
							input_gpu_address,
							maxLocation_gpu_address,
							mParameterInfoOnGPU);
						CUDA_SYNCHRONIZE_DEBUG;
					}
					else
					{
						forward_cpu_impl(input);
					}
				}

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}




			void MaxPoolingCore::backward()
			{
				if (const std::shared_ptr<TensorCore>& input_ptr = mInputTensorCoreTbl[0].lock())
				{
					auto& input = *input_ptr;


					if (input.requiresGrad())
					{
						if (m_on_cuda)
						{
							auto input_gpu_grad_address = input.getGpuGradDataAddress();
							const auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();
							const auto maxLocation_gpu_address = mMaxLocation.getGpuDataAddress();
							{
								dim3 block(32, 32);
								dim3 grid(
									(mBatchSize + block.x - 1) / block.x,
									(mIcIhIw + block.y - 1) / block.y);

								backward_gpu_impl_init << <grid, block >> > (
									input_gpu_grad_address,
									mBatchSize,
									mIcIhIw);
								CUDA_SYNCHRONIZE_DEBUG;
							}
							{
								dim3 block(32, 32);
								dim3 grid(
									(mBatchSize + block.x - 1) / block.x,
									(mOcOhOw + block.y - 1) / block.y);

								backward_gpu_impl << <grid, block >> > (
									input_gpu_grad_address,
									output_gpu_grad_address,
									maxLocation_gpu_address,
									mParameterInfoOnGPU);
								CUDA_SYNCHRONIZE_DEBUG;
							}
						}
						else
						{
							backward_cpu_impl(input);
						}

					}
				}
			}


			void MaxPoolingCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 OcOhOw = 0; OcOhOw < mOcOhOw; OcOhOw++)
					{
						const u32 Oc = OcOhOw / mOhOw;
						const u32 Ic = Oc;
						const u32 OhOw = OcOhOw - Oc * mOhOw;
						const u32 Oh = OhOw / mOw;
						const u32 Ow = OhOw - Oh * mOw;

						const s32 basisIndexIh = Oh * mSh - mPh;
						const s32 basisIndexIw = Ow * mSw - mPw;



						bool paddingIncluded = false;

						const s32 indexHStartPoint = (basisIndexIh < 0) ? (paddingIncluded = true, 0) : basisIndexIh;
						const s32 indexWStartPoint = (basisIndexIw < 0) ? (paddingIncluded = true, 0) : basisIndexIw;

						const s32 indexHEndPoint = (indexHStartPoint + mFh >= mIh) ? (paddingIncluded = true, mIh) : indexHStartPoint + mFh;
						const s32 indexWEndPoint = (indexWStartPoint + mFw >= mIw) ? (paddingIncluded = true, mIw) : indexWStartPoint + mFw;

						s32 maxIh = indexHStartPoint;
						s32 maxIw = indexWStartPoint;
						DataType maxValue = input(N, Ic, maxIh, maxIw);

						for (u32 indexH = indexHStartPoint; indexH < indexHEndPoint; indexH++)
						{
							for (u32 indexW = indexWStartPoint; indexW < indexWEndPoint; indexW++)
							{
								DataType maxCandValue = input(N, Ic, indexH, indexW);
								if (maxValue < maxCandValue)
								{
									maxValue = maxCandValue;
									maxIh = indexH;
									maxIw = indexW;
								}
							}
						}

						s32 maxValueIndex = Ic * mIhIw + maxIh * mIw + maxIw;

						if (paddingIncluded)
						{
							if (maxValue < 0.0f)
							{
								maxValue = 0.0f;
								maxValueIndex = -1;
							}
						}

						mMaxLocation[N * mOcOhOw + OcOhOw] = maxValueIndex;
						mOutput(N, OcOhOw) = maxValue;
					}
				}
			}

			void MaxPoolingCore::backward_cpu_impl(TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
					{
						input.d(N, IcIhIw) = 0;
					}

					for (u32 OcOhOw = 0; OcOhOw < mOcOhOw; OcOhOw++)
					{
						const s32 location = static_cast<s32>(mMaxLocation[N * mOcOhOw + OcOhOw]);

						//locationが0以下ならパディング領域が最大だったことを意味するため、スキップできる。
						if (location < 0)
						{
							continue;
						}

						input.d(N, location) += mOutput.d(N, OcOhOw);
					}
				}
			}
		}
	}
}