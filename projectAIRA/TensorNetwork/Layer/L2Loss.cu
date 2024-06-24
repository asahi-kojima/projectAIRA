#include "L2Loss.h"
#include "Layer.h"


namespace
{
	__global__ void forward_gpu_impl_pre(
		DataType* lossPerBatch,
		const DataType* input_lhs,
		const DataType* input_rhs,
		const u32 mBatchSize,
		const u32 mCHW)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		if (N >= mBatchSize)
		{
			return;
		}

		const u32 offset = N * mCHW;

		DataType loss = 0.0f;
		for (u32 chw = 0; chw < mCHW; chw++)
		{
			DataType value_lhs = input_lhs[offset + chw];
			DataType value_rhs = input_rhs[offset + chw];
			DataType diff = value_lhs - value_rhs;
			loss += diff * diff / 2.0f;
		}

		lossPerBatch[N] = loss / mCHW;
	}

	__global__ void forward_gpu_impl_sum(
		DataType* output,
		const DataType* lossPerBatch,
		const u32 mBatchSize)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;

		DataType sum = 0.0f;
		for (u32 N = 0; N < mBatchSize; N++)
		{
			sum += lossPerBatch[N];
		}
		output[0] = sum / mBatchSize;
	}


	__global__ void backward_gpu_impl_both(
		DataType* inputL_grad,
		DataType* inputR_grad,
		const DataType* inputL,
		const DataType* inputR,
		const u32 mBatchSize,
		const u32 mCHW)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		u32 chw = blockIdx.y * blockDim.y + threadIdx.y;
		if (N >= mBatchSize || chw >= mCHW)
		{
			return;
		}


		const u32 index = N * mCHW + chw;

		DataType valueL = inputL[index];
		DataType valueR = inputR[index];
		DataType diff = valueL - valueR;

		inputL_grad[index] = diff;
		inputR_grad[index] = -diff;
	}


	__global__ void backward_gpu_impl_oneside(
		DataType* input_grad,
		const DataType* inputFormer,
		const DataType* inputLatter,
		const u32 mBatchSize,
		const u32 mCHW)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		u32 chw = blockIdx.y * blockDim.y + threadIdx.y;
		if (N >= mBatchSize || chw >= mCHW)
		{
			return;
		}

		const u32 index = N * mCHW + chw;
		input_grad[index] = inputFormer[index] - inputLatter[index];
		
	}
	
}



namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			Layer L2Loss()
			{
				Layer l2loss = gen<L2LossCore>("L2Loss");
				return l2loss;
			}

			L2LossCore::L2LossCore()
				:BaseLayer(2, 1, 1)
				, mBatchSize(0)
				, mCHW(0)
				, mOutput(*m_output_tensorcore_tbl[0])
				, mLossPerBatch(false)
			{

			}




			BaseLayer::iotype L2LossCore::forward(const BaseLayer::iotype& input_tensors)
			{
				const auto& inputL = *getTensorCoreFrom(input_tensors[0]);
				const auto& inputR = *getTensorCoreFrom(input_tensors[1]);

				//入力間での無矛盾性のチェック
				{
					if (inputL.getBatchSize() != inputR.getBatchSize())
					{
						std::cout << "batchSize is not consistent." << std::endl;
						exit(1);
					}

					if (inputL.getCHW() != inputR.getCHW())
					{
						std::cout << "CHW Size is not consistent." << std::endl;
						exit(1);
					}
				}

				//初期化
				if (!m_init_finish)
				{
					initialize();
				}

				//出力テンソルとパラメータの形状確認＆対応
				{
					//データサイズを格納
					mBatchSize = inputL.getBatchSize();
					mCHW = inputL.getCHW();

					//出力テンソルの形状変更
					bool isInit = mOutput.reshapeAs(1, m_on_cuda);
					if (isInit)
					{
						mOutput.d(0) = 1;
					}

					//途中計算に必要なバッチ損失の形状変更
					mLossPerBatch.reshapeAs(mBatchSize, 1, m_on_cuda);
				}

				if (m_on_cuda)
				{
					auto inputL_gpu_address = inputL.getGpuDataAddress();
					auto inputR_gpu_address = inputR.getGpuDataAddress();
					auto lossPerBatch_gpu_address = mLossPerBatch.getGpuDataAddress();
					auto output_gpu_address = mOutput.getGpuDataAddress();

					{
						dim3 block(32);
						dim3 grid((mBatchSize + block.x - 1) / block.x);
						forward_gpu_impl_pre << <grid, block >> > (
							lossPerBatch_gpu_address,
							inputL_gpu_address,
							inputR_gpu_address,
							mBatchSize,
							mCHW);
						CUDA_SYNCHRONIZE_DEBUG;
					}
					{
						dim3 block(1);
						dim3 grid(1);
						forward_gpu_impl_sum << <grid, block >> > (
							output_gpu_address,
							lossPerBatch_gpu_address,
							mBatchSize);
						CUDA_SYNCHRONIZE_DEBUG;
					}

				}
				else
				{
					forward_cpu_impl(inputL, inputR);
				}

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}

			void L2LossCore::backward()
			{
				if (std::shared_ptr<TensorCore> inputL_ptr = mInputTensorCoreTbl[0].lock())
				{
					if (std::shared_ptr<TensorCore> inputR_ptr = mInputTensorCoreTbl[1].lock())
					{
						TensorCore& inputL = *inputL_ptr;
						TensorCore& inputR = *inputR_ptr;


						if (m_on_cuda)
						{
							auto inputL_gpu_grad_address = inputL.getGpuGradDataAddress();
							auto inputR_gpu_grad_address = inputR.getGpuGradDataAddress();
							auto inputL_gpu_address = inputL.getGpuDataAddress();
							auto inputR_gpu_address = inputR.getGpuDataAddress();


							bool inputL_requires_grad = inputL.requiresGrad();
							bool inputR_requires_grad = inputR.requiresGrad();

							dim3 block(32, 32);
							dim3 grid((mBatchSize + block.x - 1) / block.x, (mCHW + block.y - 1) / block.y);

							if (inputL_requires_grad && inputR_requires_grad)
							{
								backward_gpu_impl_both << <grid, block >> > (
									inputL_gpu_grad_address, 
									inputR_gpu_grad_address, 
									inputL_gpu_address,
									inputR_gpu_address,
									mBatchSize,
									mCHW);
							}
							else if (inputL_requires_grad && !inputR_requires_grad)
							{
								backward_gpu_impl_oneside << <grid, block >> > (
									inputL_gpu_grad_address,
									inputL_gpu_address,
									inputR_gpu_address,
									mBatchSize,
									mCHW);
							}
							else if (!inputL_requires_grad && inputR_requires_grad)
							{
								backward_gpu_impl_oneside << <grid, block >> > (
									inputR_gpu_grad_address,
									inputR_gpu_address,
									inputL_gpu_address,
									mBatchSize,
									mCHW);
							}
							CUDA_SYNCHRONIZE_DEBUG;
						}
						else
						{
							backward_cpu_impl(inputL, inputR);
						}

					}
					else
					{
						std::cout << "RHS resource Error@CrossEntropyWithSMCore::backward" << std::endl;
						exit(1);
					}
				}
				else
				{
					std::cout << "LHS resource Error@CrossEntropyWithSMCore::backward" << std::endl;
					exit(1);
				}
			}


			void L2LossCore::forward_cpu_impl(const TensorCore& inputL, const TensorCore& inputR)
			{
				DataType loss = 0.0f;

				for (u32 N = 0; N < mBatchSize; N++)
				{
					DataType batchLoss = 0.0f;
					for (u32 chw = 0; chw < mCHW; chw++)
					{
						DataType valueL = inputL(N, chw);
						DataType valueR = inputR(N, chw);
						DataType diff = valueL - valueR;
						batchLoss += diff * diff / 2;
					}
					loss += batchLoss / mCHW;
				}

				mOutput[0] = loss / mBatchSize;
			}

			void  L2LossCore::backward_cpu_impl(TensorCore& inputL, TensorCore& inputR)
			{
				bool inputL_need_grad = inputL.requiresGrad();
				bool inputR_need_grad = inputR.requiresGrad();

				const u32 dataSize = inputL.getDataSize();

				if (inputL_need_grad && inputR_need_grad)
				{
					for (u32 i = 0; i < dataSize; i++)
					{
						const DataType valueL = inputL[i];
						const DataType valueR = inputR[i];
						const DataType diff = valueL - valueR;
						inputL.d(i) = diff;
						inputR.d(i) = -diff;
					}
				}
				else if (inputL_need_grad && !inputR_need_grad)
				{
					for (u32 i = 0; i < dataSize; i++)
					{
						const DataType valueL = inputL[i];
						const DataType valueR = inputR[i];
						const DataType diff = valueL - valueR;
						inputL.d(i) = diff;
					}
				}
				else if (!inputL_need_grad && inputR_need_grad)
				{
					for (u32 i = 0; i < dataSize; i++)
					{
						const DataType valueL = inputL[i];
						const DataType valueR = inputR[i];
						const DataType diff = valueL - valueR;
						inputR.d(i) = -diff;
					}
				}
				else
				{
				}
			}
		}
	}
}