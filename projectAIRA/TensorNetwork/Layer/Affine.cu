#include <random>
#include "Layer.h"
#include "Affine.h"
namespace
{


	__global__ void affine_forward_gpu_impl(
		f32* y,
		const f32* x,
		const f32* A,
		const f32* b,
		const u32 batchSize,
		const u32 outputSize,
		const u32 inputSize)
	{
		u32 O = blockIdx.x * blockDim.x + threadIdx.x;
		u32 N = blockIdx.y * blockDim.y + threadIdx.y;
		if (O >= outputSize || N >= batchSize)
		{
			return;
		}

		u32 index = N * outputSize + O;

		f32 result = 0.0f;
		for (u32 I = 0; I < inputSize; I++)
		{
			result += A[O * inputSize + I] * x[N * inputSize + I];
		}

		y[index] = result + b[O];
	}

	__global__ void affine_backward_gpu_impl_input(
		DataType* dOut,
		const DataType* dIn,
		const DataType* A,
		const u32 batchSize,
		const u32 outputSize,
		const u32 inputSize)
	{
		u32 I = blockIdx.x * blockDim.x + threadIdx.x;//input
		u32 N = blockIdx.y * blockDim.y + threadIdx.y;//batch

		if (I >= inputSize || N >= batchSize)
		{
			return;
		}

		DataType result = 0.0f;
		for (u32 O = 0; O < outputSize; O++)
		{
#ifdef _DEBUG
			if (O * inputSize + I >= outputSize * inputSize)
			{
				assert(0);
			}
			if (N * outputSize + O >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += A[O * inputSize + I] * dIn[N * outputSize + O];
			//printf("A[%d, %d] = %f\n", O, I, A[O * inputSize + I]);
			//printf("DI[%d, %d] = %f\n", N, O, dIn[N * outputSize + O]);
		}
		dOut[N * inputSize + I] = result;
	}

	//Weightパラメータ
	__global__ void affine_backward_gpu_impl_weight(
		DataType* dA,
		const DataType* dout,
		const DataType* input,
		const u32 batchSize,
		const u32 outputSize,
		const u32 inputSize)
	{
		u32 I = blockIdx.x * blockDim.x + threadIdx.x;
		u32 O = blockIdx.y * blockDim.y + threadIdx.y;
		if (I >= inputSize || O >= outputSize)
		{
			return;
		}

		u32 id = O * inputSize + I;

		DataType result = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
#if INDEX_DEBUG
			if (N * inputSize + I >= batchSize * inputSize)
			{
				assert(0);
			}
			if (N * outputSize + O >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += dout[N * outputSize + O] * input[N * inputSize + I];
		}

		dA[id] = result;
	}

	//Biasパラメータ
	__global__ void affine_backward_gpu_impl_bias(
		DataType* dBias,
		const DataType* output_grad,
		const u32 batchSize,
		const u32 outputSize)
	{
		u32 O = blockIdx.x * blockDim.x + threadIdx.x;
		if (O >= outputSize)
		{
			return;
		}

		DataType result = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
#if INDEX_DEBUG
			if ((N * outputSize + O) >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += output_grad[N * outputSize + O];
		}
#if INDEX_DEBUG
		if (O >= outputSize)
		{
			assert(0);
		}
#endif
		dBias[O] = result;
	}
}




namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer Affine(u32 output_size)
			{
				Layer affine = gen<AffineCore>("Affine", output_size);
				return affine;
			}




			AffineCore::AffineCore(u32 output_size, DataType affineWeight)
				:BaseLayer(1, 1, 1, 2)
				, mAffineWeight(affineWeight)
				, m_output_size(output_size)
				, m_batch_size(0)
				, m_input_size(0)
				, mOutput(*m_output_tensorcore_tbl[0])
				, mWeight(*mTrainableParameterTbl[0])
				, mBias(*mTrainableParameterTbl[1])
			{
			}

			AffineCore::~AffineCore()
			{}

			BaseLayer::iotype  AffineCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}


				const auto& input = *getTensorCoreFrom(input_tensors[0]);

				const u32  input_batchSize = input.getBatchSize();
				const u32  input_chw = input.getCHW();;
				const bool input_on_cuda = input.isOnCuda();


				//出力テンソルと訓練パラメータの形状確認＆対応
				{
					m_batch_size = input_batchSize;
					m_input_size = input_chw;
					//m_on_cuda = input_on_cuda;

					//出力テンソルの形状変更
					mOutput.reshapeAs(input_batchSize, m_output_size, input_on_cuda);

					//weightの形状変更
					bool isWeightInit = mWeight.reshapeAs(m_output_size, m_input_size, input_on_cuda);

					//biasの形状変更
					//バイアスは初回だけ初期化され、それ以降は変化しない。
					bool isBiasInit = mBias.reshapeAs(m_output_size, input_on_cuda);



					if (isWeightInit)
					{
#ifdef _DEBUG
						std::cout << "Weight Param was initialized." << std::endl;
#endif // _DEBUG
						std::random_device seed_gen;
						std::default_random_engine engine(seed_gen());
						std::normal_distribution<> dist(0.0f, std::sqrt(2.0f / m_input_size));
						for (u32 i = 0, end = mWeight.getDataSize(); i < end; i++)
						{
							mWeight[i] = mAffineWeight * static_cast<DataType>(dist(engine));
						}
						mWeight.synchronize_from_CPU_to_GPU();
					}

					if (isBiasInit)
					{
#ifdef _DEBUG
						std::cout << "Bias Param was initialized." << std::endl;
#endif // _DEBUG
						for (u32 i = 0, end = mBias.getDataSize(); i < end; i++)
						{
							mBias[i] = 0.0f;
						}
						mBias.synchronize_from_CPU_to_GPU();
					}
				}


				//順伝搬処理
				{
					if (m_on_cuda)
					{
						auto output_gpu_address = mOutput.getGpuDataAddress();
						auto input_gpu_address = input.getGpuDataAddress();
						auto weight_gpu_address = mWeight.getGpuDataAddress();
						auto bias_gpu_address = mBias.getGpuDataAddress();

						dim3 block(32, 32);
						dim3 grid((m_output_size + block.x - 1) / block.x, (m_batch_size + block.y - 1) / block.y);
						affine_forward_gpu_impl << <grid, block >> > (
							output_gpu_address,
							input_gpu_address,
							weight_gpu_address,
							bias_gpu_address,
							m_batch_size,
							m_output_size,
							m_input_size);
						CUDA_SYNCHRONIZE_DEBUG;
					}
					else
					{
						forward_cpu_impl(input);
					}
				}


				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}


			void AffineCore::backward()
			{
				//std::cout << "Affine Backward" << std::endl;
				if (const std::shared_ptr<TensorCore>& input_ptr = mInputTensorCoreTbl[0].lock())
				{
					auto& input = *input_ptr;

					auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();

					auto input_gpu_address = input.getGpuDataAddress();
					auto input_gpu_grad_address = input.getGpuGradDataAddress();

					auto weight_gpu_address = mWeight.getGpuDataAddress();
					auto weight_gpu_grad_address = mWeight.getGpuGradDataAddress();

					auto bias_gpu_grad_address = mBias.getGpuGradDataAddress();

					//パラメータの逆伝搬
					{
						if (m_on_cuda)
						{

							//Weight
							{
								dim3 block(16, 16);
								dim3 grid((m_input_size + block.x - 1) / block.x, (m_output_size + block.y - 1) / block.y);
								affine_backward_gpu_impl_weight << <grid, block >> > (
									weight_gpu_grad_address,
									output_gpu_grad_address,
									input_gpu_address,
									m_batch_size,
									m_output_size,
									m_input_size);
								CUDA_SYNCHRONIZE_DEBUG;
							}

							//Bias
							{
								dim3 block(16);
								dim3 grid((m_output_size + block.x - 1) / block.x);
								affine_backward_gpu_impl_bias << <grid, block >> > (
									bias_gpu_grad_address,
									output_gpu_grad_address,
									m_batch_size,
									m_output_size);
								CUDA_SYNCHRONIZE_DEBUG;
							}
						}
						else
						{
							backward_cpu_impl_parameter(input);
						}
					}

					if (input.requiresGrad())//勾配不要の場合、逆伝搬はスキップ出来る。
					{
						if (m_on_cuda)
						{
							dim3 block(16, 16);
							dim3 grid((m_input_size + block.x - 1) / block.x, (m_batch_size + block.y - 1) / block.y);
							affine_backward_gpu_impl_input << <grid, block >> > (
								input_gpu_grad_address,
								output_gpu_grad_address,
								weight_gpu_address,
								m_batch_size,
								m_output_size,
								m_input_size);
							CUDA_SYNCHRONIZE_DEBUG;
						}
						else
						{
							backward_cpu_impl_input(input);
						}
					}
				}
				else
				{
					std::cout << "Resource0 Error@ReLUCore::backward" << std::endl;
					exit(1);
				}
			}



			void AffineCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 N = 0, end = mOutput.getBatchSize(); N < end; N++)
				{
					for (u32 O = 0; O < m_output_size; O++)
					{
						DataType result = 0.0f;
						for (u32 I = 0; I < m_input_size; I++)
						{
							result += mWeight(O, I) * input(N, I);
						}

						mOutput(N, O) = result + mBias(O);
					}
				}
			}

			void AffineCore::backward_cpu_impl_input(TensorCore& input)
			{
				for (u32 N = 0; N < m_batch_size; N++)
				{
					for (u32 I = 0; I < m_input_size; I++)
					{
						DataType result = 0.0f;
						for (u32 O = 0; O < m_output_size; O++)
						{
							result += mWeight(O, I) * mOutput.d(N, O);
						}
						input.d(N, I) = result;
					}
				}
			}

			void AffineCore::backward_cpu_impl_parameter(const TensorCore& input)
			{
				for (u32 O = 0; O < m_output_size; O++)
				{
					for (u32 I = 0; I < m_input_size; I++)
					{
						DataType result = 0.0f;
						for (u32 N = 0; N < m_batch_size; N++)
						{
							result += input(N, I) * mOutput.d(N, O);
						}
						mWeight.d(O, I) = result;
					}

					DataType result = 0.0f;
					for (u32 N = 0; N < m_batch_size; N++)
					{
						result += mOutput.d(N, O);
					}
					mBias.d(O) = result;
				}
			}


		}
	}
}