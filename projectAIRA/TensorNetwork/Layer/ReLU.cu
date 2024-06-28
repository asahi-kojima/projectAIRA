#include "ReLU.h"
#include "Layer.h"



namespace
{
	__global__ void forward_impl_gpu(DataType* output_ptr, const DataType* input_ptr, DataType* mask_ptr, u32 dataSize)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dataSize)
		{
			return;
		}

		const DataType& input_value = input_ptr[i];
		auto& mask = mask_ptr[i];

		if (input_value > 0)
			mask = 1;
		else
			mask = 0;

		output_ptr[i] = input_value * mask;
	}


	__global__ void backward_impl_gpu(DataType* doutput_ptr, const DataType* dinput_ptr, const DataType* mask_ptr, u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		doutput_ptr[i] = dinput_ptr[i] * mask_ptr[i];
	}
}


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			Layer aoba::nn::layer::ReLU()
			{
				Layer relu = gen<ReLUCore>("Add");
				return relu;
			}

			ReLUCore::ReLUCore()
				: BaseLayer(1, 1, 1)
				, mDataSize(0)
				, mOutput(*m_output_tensorcore_tbl[0])
				, mMask(false)
			{
			}


			BaseLayer::iotype ReLUCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}

				const auto& input = *getTensorCoreFrom(input_tensors[0]);



				//�o�̓e���\���ƃp�����[�^�̌`��m�F���Ή�
				{
					//�f�[�^�T�C�Y���i�[
					mDataSize = input.getDataSize();
					//m_on_cuda = input.m_on_cuda;

					//�o�̓e���\���̌`��ύX
					mOutput.reshapeAs(input, input.isOnCuda());

					//�}�X�N�̌`��ύX
					mMask.reshapeAs(input, input.isOnCuda());
				}




				if (m_on_cuda)
				{
					auto output_gpu_address = mOutput.getGpuDataAddress();
					auto input_gpu_address = input.getGpuDataAddress();
					auto mask_gpu_address = mMask.getGpuDataAddress();

					dim3 block(256);
					dim3 grid((mDataSize + block.x - 1) / block.x);
#ifdef TIME_DEBUG
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
					forward_impl_gpu << <grid, block >> > (output_gpu_address, input_gpu_address, mask_gpu_address, mDataSize);
					CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_impl_gpu");
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

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}


			void ReLUCore::backward()
			{
				if (const std::shared_ptr<TensorCore>& input_tensor_core = mInputTensorCoreTbl[0].lock())
				{
					TensorCore& input = *input_tensor_core;
					if (input.requiresGrad())
					{
						auto& input_tensorcore = *input_tensor_core;
						if (m_on_cuda)
						{
							auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();
							auto input_gpu_grad_address = input_tensorcore.getGpuGradDataAddress();
							auto mask_gpu_address = mMask.getGpuDataAddress();

							dim3 block(32);
							dim3 grid((mDataSize + block.x - 1) / block.x);
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_impl_gpu << <grid, block >> > (input_gpu_grad_address, output_gpu_grad_address, mask_gpu_address, mDataSize);
							CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_impl_gpu");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
						else
						{
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_cpu_impl(*input_tensor_core);
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


			void ReLUCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 i = 0; i < mDataSize; i++)
				{
					const auto& input_value = input[i];
					auto& mask = mMask[i];

					if (input_value > 0)
						mask = 1;
					else
						mask = 0;

					mOutput[i] = input_value * mask;
				}
			}
			void ReLUCore::backward_cpu_impl(TensorCore& input)
			{
				for (u32 i = 0; i < mDataSize; i++)
				{
					input.d(i) = mOutput.d(i) * mMask[i];
				}
			}

		}
	}
}