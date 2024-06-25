#include "Layer.h"
#include "BatchNorm.h"

namespace
{
	constexpr DataType EP = 1e-7;



	__global__ void forward_gpu_impl_computeBlockMeans(
		DataType* blockMean,
		DataType* blockSqMean,
		const DataType* input,
		const u32 mBatchSize,
		const u32 mIc,
		const u32 mIhIw)
	{
		const u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 N = blockIdx.y * blockDim.y + threadIdx.y;

		if (Ic >= mIc || N >= mBatchSize)
		{
			return;
		}


		const u32 mIcIhIw = mIc * mIhIw;

		DataType mean = 0.0f;
		DataType sqMean = 0.0f;

		//------------------------------------------------------------------
		//平均を計算
		//------------------------------------------------------------------
		for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
		{
			DataType value = input[N * mIcIhIw + Ic * mIhIw + IhIw];
			mean += value;
			sqMean += value * value;
		}

		blockMean[N * mIc + Ic] = mean;
		blockSqMean[N * mIc + Ic] = sqMean;
	}


	__global__ void forward_gpu_impl_computeMeanSigma(
		DataType* Mean,
		DataType* Sigma,
		const DataType* blockMean,
		const DataType* blockSqMean,
		const u32 mBatchSize,
		const u32 mIc,
		const u32 mIhIw)
	{
		u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;

		if (Ic >= mIc)
		{
			return;
		}


		DataType mean = 0.0f;
		DataType sqMean = 0.0f;

		//------------------------------------------------------------------
		//平均を計算
		//------------------------------------------------------------------
		for (u32 N = 0; N < mBatchSize; N++)
		{
			mean += blockMean[N * mIc + Ic];
			sqMean += blockSqMean[N * mIc + Ic];
		}

		mean /= (mBatchSize * mIhIw);
		sqMean /= (mBatchSize * mIhIw);

		//------------------------------------------------------------------
		//偏差を計算
		//------------------------------------------------------------------
		Mean[Ic] = mean;
		Sigma[Ic] = std::sqrt(sqMean - mean * mean) + EP;
	}


	__global__ void forward_gpu_impl_main(
		DataType* output,
		DataType* intermediateResult,
		const DataType* input,
		const DataType* Gamma,
		const DataType* Beta,
		const DataType* Mean,
		const DataType* Sigma,
		u32 mBatchSize,
		u32 mIc,
		u32 mIhIw)
	{
		const u32 IhIw = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 Ic = blockIdx.y * blockDim.y + threadIdx.y;
		const u32 N = blockIdx.z * blockDim.z + threadIdx.z;
		if (IhIw >= mIhIw || Ic >= mIc || N >= mBatchSize)
		{
			return;
		}
		

		u32 mIcIhIw = mIc * mIhIw;

		DataType mean = Mean[Ic];
		DataType sigma = Sigma[Ic];

		//------------------------------------------------------------------
		//標準化
		//------------------------------------------------------------------
		DataType gamma = Gamma[Ic];
		DataType beta = Beta[Ic];


		u32 index = N * mIcIhIw + Ic * mIhIw + IhIw;
		DataType normalizeResult = (input[index] - mean) / sigma;

		intermediateResult[index] = normalizeResult;
		output[index] = gamma * normalizeResult + beta;
	}


	__global__ void backward_gpu_impl_computeBlock(
		DataType* blockGamma_grad,
		DataType* blockBeta_grad,
		const DataType* output_grad,
		const DataType* intermediateResult,
		const u32 mBatchSize,
		const u32 mIc,
		const u32 mIhIw)
	{
		u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;
		u32 N = blockIdx.y * blockDim.y + threadIdx.y;

		if (Ic >= mIc || N >= mBatchSize)
		{
			return;
		}

		u32 mIcIhIw = mIc * mIhIw;

		DataType dGamma = 0.0f;
		DataType dBeta = 0.0f;

		for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
		{
			u32 index = N * mIcIhIw + Ic * mIhIw + IhIw;
			DataType dO = output_grad[index];
			DataType iR = intermediateResult[index];

			dGamma += dO * iR;
			dBeta += dO;
		}
		
		blockGamma_grad[N * mIc + Ic] = dGamma;
		blockBeta_grad[N * mIc + Ic] = dBeta;
	}

	__global__ void backward_gpu_impl_computeParamGrad(
		DataType* gamma_grad,
		DataType* beta_grad,
		const DataType* blockGamma_grad,
		const DataType* blockBeta_grad,
		const u32 mBatchSize,
		const u32 mIc,
		const u32 mIhIw)
	{
		const u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;

		if (Ic >= mIc)
		{
			return;
		}

		const u32 mIcIhIw = mIc * mIhIw;


		DataType gamma = 0.0f;
		DataType beta = 0.0f;

		for (u32 N = 0; N < mBatchSize; N++)
		{
			gamma += blockGamma_grad[N * mIc + Ic];
			beta += blockBeta_grad[N * mIc + Ic];
		}

		gamma_grad[Ic] = gamma;
		beta_grad[Ic] = beta;
	}

	__global__ void backward_gpu_impl_main(
		DataType* input_grad,
		const DataType* intermediateResult,
		const DataType* output_grad,
		const DataType* gamma,
		const DataType* gamma_grad,
		const DataType* beta_grad,
		const DataType* sigma,
		const u32 mBatchSize,
		const u32 mIc,
		const u32 mIhIw)
	{
		const u32 IhIw = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 Ic = blockIdx.y * blockDim.y + threadIdx.y;
		const u32 N = blockIdx.z * blockDim.z + threadIdx.z;

		const u32 mIcIhIw = mIc * mIhIw;
		if (N >= mBatchSize || Ic >= mIc || IhIw >= mIhIw)
		{
			return;
		}
	

		const u32 index = N * mIcIhIw + Ic * mIhIw + IhIw;

		const DataType diMean = gamma_grad[Ic] / (mBatchSize * mIhIw);
		const DataType dMean = beta_grad[Ic] / (mBatchSize * mIhIw);

		input_grad[index]
			=
			(gamma[Ic] / sigma[Ic]) * (output_grad[index] - dMean - intermediateResult[index] * diMean);
	}
}



namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			Layer BatchNorm()
			{
				Layer batchNorm = gen<BatchNormCore>("BatchNorm");
				return batchNorm;
			}


			BatchNormCore::BatchNormCore()
				:BaseLayer(1, 1, 1, 2)
				, mOutput(*m_output_tensorcore_tbl[0])
				, mGamma(*mTrainableParameterTbl[0])
				, mBeta(*mTrainableParameterTbl[1])
				, mIntermediate(false)
				, mSigma(false)
				, mMean(false)

				, mBlockMean(true)
				, mBlockSqMean(true)
			{}

			BatchNormCore::~BatchNormCore()
			{}

			BaseLayer::iotype  BatchNormCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}

				const auto& input = *getTensorCoreFrom(input_tensors[0]);
				{
					mBatchSize = input.getBatchSize();
					mIc = input.getChannel();
					mIhIw = input.getHW();
					mIcIhIw = input.getCHW();


					//形状変形
					mOutput.reshapeAs(input, m_on_cuda);

					mIntermediate.reshapeAs(input, m_on_cuda);

					bool isGammaInit = mGamma.reshapeAs(mIc, m_on_cuda);
					bool isBetaInit = mBeta.reshapeAs(mIc, m_on_cuda);

					mSigma.reshapeAs(mIc, m_on_cuda);

					if (m_on_cuda)
					{
						mMean.reshapeAs(mIc, m_on_cuda);
						mBlockMean.reshapeAs(mBatchSize, mIc, m_on_cuda);
						mBlockSqMean.reshapeAs(mBatchSize, mIc, m_on_cuda);
					}

					//変形後の初期化部分
					if (isGammaInit)
					{
#ifdef _DEBUG
						std::cout << "mGamma Param was initialized." << std::endl;
#endif // _DEBUG
						for (u32 i = 0, end = mGamma.getDataSize(); i < end; i++)
						{
							mGamma[i] = 1.0f;
						}
						mGamma.synchronize_from_CPU_to_GPU();
					}

					if (isBetaInit)
					{
#ifdef _DEBUG
						std::cout << "mBeta Param was initialized." << std::endl;
#endif // _DEBUG
						for (u32 i = 0, end = mBeta.getDataSize(); i < end; i++)
						{
							mBeta[i] = 0.0f;
						}
						mBeta.synchronize_from_CPU_to_GPU();
					}
				}

				//順伝搬処理
				{
					if (m_on_cuda)
					{
						auto output_gpu_address = mOutput.getGpuDataAddress();
						auto intermediate_gpu_address = mIntermediate.getGpuDataAddress();
						const auto input_gpu_address = input.getGpuDataAddress();

						auto blockMean_gpu_address = mBlockMean.getGpuDataAddress();
						auto blockSqMean_gpu_address = mBlockSqMean.getGpuDataAddress();

						auto mean_gpu_address = mMean.getGpuDataAddress();
						auto sigma_gpu_address = mSigma.getGpuDataAddress();

						const auto gamma_gpu_address = mGamma.getGpuDataAddress();
						const auto beta_gpu_address = mBeta.getGpuDataAddress();

						{
							dim3 block(16, 16);
							dim3 grid(
								(mIc + block.x - 1) / block.x,
								(mBatchSize + block.y - 1) / block.y);
							forward_gpu_impl_computeBlockMeans << <grid, block >> > (
								blockMean_gpu_address,
								blockSqMean_gpu_address,
								input_gpu_address,
								mBatchSize,
								mIc,
								mIhIw);
							CUDA_SYNCHRONIZE_DEBUG;
						}
						{
							dim3 block(16);
							dim3 grid(
								(mIc + block.x - 1) / block.x);
							forward_gpu_impl_computeMeanSigma << <grid, block >> >
								(
									mean_gpu_address,
									sigma_gpu_address,
									blockMean_gpu_address,
									blockSqMean_gpu_address,
									mBatchSize,
									mIc,
									mIhIw);
							CUDA_SYNCHRONIZE_DEBUG;
						}
						{
							dim3 block(16, 8, 8);//ok
							dim3 grid(
								(mIhIw + block.x - 1) / block.x,
								(mIc + block.y - 1) / block.y,
								(mBatchSize + block.z - 1) / block.z);
							forward_gpu_impl_main << <grid, block >> > (
								output_gpu_address,
								intermediate_gpu_address,
								input_gpu_address,
								gamma_gpu_address,
								beta_gpu_address,
								mean_gpu_address,
								sigma_gpu_address,
								mBatchSize,
								mIc,
								mIhIw);
							CUDA_SYNCHRONIZE_DEBUG;
						}
					}
					else
					{
						forward_cpu_impl(input);
					}
				}


				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}


			void BatchNormCore::backward()
			{
				if (const std::shared_ptr<TensorCore>& input_ptr = mInputTensorCoreTbl[0].lock())
				{
					auto& input = *input_ptr;

					if (m_on_cuda)
					{
						const auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();
						const auto intermediate_gpu_address = mIntermediate.getGpuDataAddress();
						auto input_gpu_grad_address = input.getGpuGradDataAddress();

						auto gamma_gpu_address = mGamma.getGpuDataAddress();
						auto gamma_gpu_grad_address = mGamma.getGpuGradDataAddress();
						auto beta_gpu_grad_address = mBeta.getGpuGradDataAddress();
						const auto sigma_gpu_address = mSigma.getGpuDataAddress();

						auto blockGamma_gpu_grad_address = mBlockSqMean.getGpuGradDataAddress();
						auto blockBeta_gpu_grad_address = mBlockMean.getGpuGradDataAddress();

						//ブロック計算
						{
							dim3 block(32, 32);//OK
							dim3 grid(
								(mIc + block.x - 1) / block.x, 
								(mBatchSize + block.y - 1) / block.y);
							backward_gpu_impl_computeBlock << <grid, block >> > (
								blockGamma_gpu_grad_address,
								blockBeta_gpu_grad_address,
								output_gpu_grad_address,
								intermediate_gpu_address,
								mBatchSize,
								mIc,
								mIhIw);
							CUDA_SYNCHRONIZE_DEBUG;
						}
						//パラメータの勾配計算
						{
							dim3 block(32);
							dim3 grid((mIc + block.x - 1) / block.x);
							backward_gpu_impl_computeParamGrad << <grid, block >> > (
								gamma_gpu_grad_address,
								beta_gpu_grad_address,
								blockGamma_gpu_grad_address,
								blockBeta_gpu_grad_address,
								mBatchSize,
								mIc,
								mIhIw);
							CUDA_SYNCHRONIZE_DEBUG;
						}

						if (input.requiresGrad())
						{
							dim3 block(16, 8, 8);//OK
							dim3 grid(
								(mIhIw + block.x - 1) / block.x,
								(mIc + block.y - 1) / block.y,
								(mBatchSize + block.z - 1) / block.z);
							backward_gpu_impl_main << <grid, block >> > (
								input_gpu_grad_address,
								intermediate_gpu_address,
								output_gpu_grad_address,
								gamma_gpu_address,
								gamma_gpu_grad_address,
								beta_gpu_grad_address,
								sigma_gpu_address,
								mBatchSize,
								mIc,
								mIhIw);
								CUDA_SYNCHRONIZE_DEBUG;
						}

					}
					else
					{
						backward_cpu_impl(input);
					}
				}
				else
				{
					std::cout << "Resource0 Error@ReLUCore::backward" << std::endl;
					exit(1);
				}
			}




			void BatchNormCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 Ic = 0; Ic < mIc; Ic++)
				{
					DataType mean = 0.0f;
					DataType sqMean = 0.0f;
					DataType sigma = 0.0f;

					//------------------------------------------------------------------
					//平均を計算
					//------------------------------------------------------------------
					for (u32 N = 0; N < mBatchSize; N++)
					{
						for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
						{
							DataType value = input(N, Ic, IhIw);
							mean += value;
							sqMean += value * value;
						}
					}
					mean /= (mBatchSize * mIhIw);
					sqMean /= (mBatchSize * mIhIw);

					//------------------------------------------------------------------
					//偏差を計算
					//------------------------------------------------------------------
					sigma = std::sqrt(sqMean - mean * mean) + EP;
					mSigma[Ic] = sigma;

					//------------------------------------------------------------------
					//標準化
					//------------------------------------------------------------------
					DataType gamma = mGamma[Ic];
					DataType beta = mBeta[Ic];
					for (u32 N = 0; N < mBatchSize; N++)
					{
						for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
						{
							DataType normalizeResult = (input(N, Ic, IhIw) - mean) / sigma;
							mIntermediate(N, Ic, IhIw) = normalizeResult;
							mOutput(N, Ic, IhIw) = gamma * normalizeResult + beta;
						}
					}

				}
			}

			void BatchNormCore::backward_cpu_impl(TensorCore& input)
			{
				for (u32 Ic = 0; Ic < mIc; Ic++)
				{
					DataType dGamma = 0.0f;
					DataType dBeta = 0.0f;
					for (u32 N = 0; N < mBatchSize; N++)
					{
						for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
						{
							DataType output_grad = mOutput.d(N, Ic, IhIw);
							dGamma += output_grad * mIntermediate(N, Ic, IhIw);
							dBeta += output_grad;
						}
					}
					mGamma.d(Ic) = dGamma;
					mBeta.d(Ic) = dBeta;


					if (input.requiresGrad())
					{
						DataType diMean = dGamma / (mBatchSize * mIhIw);
						DataType dMean = dBeta / (mBatchSize * mIhIw);

						for (u32 N = 0; N < mBatchSize; N++)
						{
							for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
							{
								input.d(N, Ic, IhIw)
									=
									(mGamma[Ic] / (mSigma[Ic] + EP)) * (mOutput.d(N, Ic, IhIw) - dMean - mIntermediate(N, Ic, IhIw) * diMean);
							}
						}
					}
				}
			}
		}
	}
}