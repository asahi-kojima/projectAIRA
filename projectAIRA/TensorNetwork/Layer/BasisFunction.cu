#include "BasisFunction.h"


namespace
{
	//TanhŠÖŒW
	__host__ __device__ DataType hd_tanh_forward(DataType x)
	{
		return tanh(x);
	}
	__global__ void g_tanh_forward(DataType* output, const DataType* input, u32 dataSize)
	{
		const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		output[index] = hd_tanh_forward(input[index]);
	}

	__host__ __device__ DataType hd_tanh_backward(DataType output_grad, DataType input)
	{
		const DataType cosh_value = cosh(input);
		return output_grad * (1 / (cosh_value * cosh_value));
	}
	__global__ void g_tanh_backward(DataType* input_grad, const DataType* output_grad, const DataType* input, u32 dataSize)
	{
		const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		const DataType output_grad_value = output_grad[index];
		const DataType input_value = input[index];
		input_grad[index] = hd_tanh_backward(output_grad_value, input_value);
	}


	//SigmoidŠÖŒW
	__host__ __device__ DataType hd_sigmoid_forward(DataType x)
	{
		return 1 / (1 + exp(-x));
	}
	__global__ void g_sigmoid_forward(DataType* output, const DataType* input, u32 dataSize)
	{
		const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		output[index] = hd_sigmoid_forward(input[index]);
	}

	__host__ __device__ DataType hd_sigmoid_backward(DataType output_grad, DataType input)
	{
		const DataType exp_ninus_x = exp(-input);
		return output_grad * (exp_ninus_x / ((1+ exp_ninus_x) * (1+ exp_ninus_x)));
	}
	__global__ void g_sigmoid_backward(DataType* input_grad, const DataType* output_grad, const DataType* input, u32 dataSize)
	{
		const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		const DataType output_grad_value = output_grad[index];
		const DataType input_value = input[index];
		input_grad[index] = hd_sigmoid_backward(output_grad_value, input_value);
	}
}

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer Tanh()
			{
				Layer tanh = gen<BasisFunctionCore>("Tanh", hd_tanh_forward, g_tanh_forward, hd_tanh_backward, g_tanh_backward);
				return tanh;
			}

			Layer Sigmoid()
			{
				Layer sigmoid = gen<BasisFunctionCore>("Sigmoid", hd_sigmoid_forward, g_sigmoid_forward, hd_sigmoid_backward, g_sigmoid_backward);
				return sigmoid;
			}



			BasisFunctionCore::BasisFunctionCore(
				CPUFunctionF functionCPU_forward, GPUFunctionF functionGPU_forward,
				CPUFunctionB functionCPU_backward, GPUFunctionB functionGPU_backward)
				:BaseLayer(1,1,1)
				,mOutput(*m_output_tensorcore_tbl[0])
				,mDataSize(0)
				,mFunctionCPU_forward(functionCPU_forward)
				,mFunctionGPU_forward(functionGPU_forward)
				,mFunctionCPU_backward(functionCPU_backward)
				,mFunctionGPU_backward(functionGPU_backward)
			{

			}

			

			BasisFunctionCore::~BasisFunctionCore()
			{

			}

			BaseLayer::iotype BasisFunctionCore::forward(const iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}

				const auto& input = *getTensorCoreFrom(input_tensors[0]);

				{
					mDataSize = input.getDataSize();

					mOutput.reshapeAs(input, m_on_cuda);
				}


				if (m_on_cuda)
				{
					auto output_gpu_address = mOutput.getGpuDataAddress();
					auto input_gpu_address = input.getGpuDataAddress();
					dim3 block(128);
					dim3 grid((mDataSize + block.x - 1) / block.x);
#ifdef TIME_DEBUG
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
					mFunctionGPU_forward << <grid, block >> > (output_gpu_address, input_gpu_address, mDataSize);
					CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "mFunctionGPU_forward");
					debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
				}
				else
				{
#ifdef TIME_DEBUG
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
					forward_cpu_impl(input);
#ifdef TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_cpu_impl");
					debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
				}


				return iotype{ m_output_tensorcore_tbl[0] };
			}

			void BasisFunctionCore::backward()
			{
				if (const std::shared_ptr<TensorCore>& input_ptr = mInputTensorCoreTbl[0].lock())
				{
					TensorCore& input = *input_ptr;
					if (input.requiresGrad())
					{
						if (m_on_cuda)
						{
							auto input_gpu_grad_address = input.getGpuGradDataAddress();
							const auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();
							const auto input_gpu_address = input.getGpuDataAddress();

							dim3 block(32);
							dim3 grid((mDataSize + block.x - 1) / block.x);
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							mFunctionGPU_backward << <grid, block >> > (
								input_gpu_grad_address, 
								output_gpu_grad_address,
								input_gpu_address,
								mDataSize);
							CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "mFunctionGPU_backward");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
						else
						{
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_cpu_impl(input);
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_cpu_impl");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
					}
				}
				else
				{
					std::cout << "Resource Error@ReLUCore::backward" << std::endl;
					exit(1);
				}
			}

			void BasisFunctionCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 i = 0; i < mDataSize; i++)
				{
					const auto input_value = input[i];
					mOutput[i] = mFunctionCPU_forward(input_value);
				}
			}
			void BasisFunctionCore::backward_cpu_impl(TensorCore& input)
			{
				for (u32 i = 0; i < mDataSize; i++)
				{
					const auto output_grad_value = mOutput.d(i);
					const auto input_value = input(i);
					input.d(i) = mFunctionCPU_backward(output_grad_value, input_value);
				}
			}
		}
	}
}