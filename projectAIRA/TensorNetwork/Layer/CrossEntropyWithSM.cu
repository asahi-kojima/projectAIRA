#include "CrossEntropyWithSM.h"
#include "Layer.h"

namespace
{
	__global__ void forward_gpu_impl_pre(
		DataType* lossPerBatch,
		const DataType* inference,
		const DataType* correct,
		const u32 batchSize,
		const u32 label_num)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		if (N >= batchSize)
		{
			return;
		}

		const u32 offset = N * label_num;

		DataType max = inference[offset + 0];
		for (u32 i = 0; i < label_num; i++)
		{
			DataType cand = inference[offset + i];
			if (max < cand)
			{
				max = cand;
			}
		}

		DataType sum = 0.0f;
		for (u32 i = 0; i < label_num; i++)
		{
			DataType value = inference[offset + i] - max;
			sum += std::exp(value);
		}

		u32 correct_index = static_cast<u32>(correct[N]);
		if (correct_index >= label_num)
		{
			assert(0);
		}

		DataType value = std::exp(inference[offset + correct_index] - max) / sum;
		DataType loss = -std::log(value + 1e-7);

		lossPerBatch[N] = loss;
	}

	__global__ void forward_gpu_impl_sum(
		DataType* output,
		const DataType* lossPerBatch,
		const u32 batchSize)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		DataType sum = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
			sum += lossPerBatch[N];
		}
		output[0] = sum / batchSize;
	}

	__global__ void backward_gpu_impl(
		DataType* d_inference,
		const DataType* inference,
		const DataType* correct,
		const u32 batchSize,
		const u32 label_num)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		if (N >= batchSize)
		{
			return;
		}

		const u32 offset = N * label_num;

		DataType max = inference[offset + 0];
		for (u32 i = 0; i < label_num; i++)
		{
			DataType cand = inference[offset + i];
			if (max < cand)
			{
				max = cand;
			}
		}

		DataType sum = 0.0f;
		for (u32 i = 0; i < label_num; i++)
		{
			DataType value = inference[offset + i] - max;
			sum += std::exp(value);
		}

		const u32 correct_label = static_cast<u32>(correct[N]);
		if (correct_label >= label_num)
		{
			assert(0);
		}

		for (u32 I = 0; I < label_num; I++)
		{
			DataType value = std::exp(inference[offset + I] - max) / sum;
			d_inference[offset + I] = value - (correct_label == I ? 1 : 0);
		}
	}
}

namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer CrossEntropyWithSM()
			{
				Layer add_layer = gen<CrossEntropyWithSMCore>("CrossEntropyWithSM");
				return add_layer;
			}


			CrossEntropyWithSMCore::CrossEntropyWithSMCore()
				: BaseLayer(2, 1, 1)
				, m_batch_size(0)
				, m_label_num(0)
				, mOutput(*m_output_tensorcore_tbl[0])
				, mLossPerBatch(false)
			{
			}



			BaseLayer::iotype CrossEntropyWithSMCore::forward(const BaseLayer::iotype& input_tensors)
			{


				const auto& inference = *getTensorCoreFrom(input_tensors[0]);
				const auto& correct = *getTensorCoreFrom(input_tensors[1]);


				//���͊Ԃł̖��������̃`�F�b�N
				{
					if (inference.getBatchSize() != correct.getBatchSize())
					{
						std::cout << "batchSize is not consistent." << std::endl;
						exit(1);
					}
				}

				//������
				if (!m_init_finish)
				{
					initialize();
				}

				//�o�̓e���\���ƃp�����[�^�̌`��m�F���Ή�
				{
					//�f�[�^�T�C�Y���i�[
					m_batch_size = inference.getBatchSize();
					m_label_num = inference.getCHW();

					//�o�̓e���\���̌`��ύX
					bool isInit = mOutput.reshapeAs(1, inference.isOnCuda());
					if (isInit)
					{
						mOutput.d(0) = 1;
					}

					//�r���v�Z�ɕK�v�ȃo�b�`�����̌`��ύX
					mLossPerBatch.reshapeAs(m_batch_size, 1, inference.isOnCuda());
				}





				if (m_on_cuda)
				{
					auto inference_gpu_address = inference.getGpuDataAddress();
					auto correct_gpu_address = correct.getGpuDataAddress();
					auto lossPerBatch_gpu_address = mLossPerBatch.getGpuDataAddress();
					auto output_gpu_address = mOutput.getGpuDataAddress();

					{
						dim3 block(32);
						dim3 grid((m_batch_size + block.x - 1) / block.x);
#ifdef TIME_DEBUG
						std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
						forward_gpu_impl_pre << <grid, block >> > (
							lossPerBatch_gpu_address, 
							inference_gpu_address, 
							correct_gpu_address, 
							m_batch_size, 
							m_label_num);
						CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
						f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
						std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_gpu_impl_pre");
						debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
					}
					{
						dim3 block(1);
						dim3 grid(1);
#ifdef TIME_DEBUG
						std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
						forward_gpu_impl_sum << <grid, block >> > (
							output_gpu_address, 
							lossPerBatch_gpu_address, 
							m_batch_size);
						CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
						f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
						std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_gpu_impl_sum");
						debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
					}
				}
				else
				{
#ifdef TIME_DEBUG
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
					forward_cpu_impl(inference, correct);
#ifdef TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_cpu_impl");
					debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
				}

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}



			void CrossEntropyWithSMCore::backward()
			{
				//std::cout << "CrossEntropyWithSM backward" << std::endl;
				if (std::shared_ptr<TensorCore> inference_ptr = mInputTensorCoreTbl[0].lock())
				{
					if (std::shared_ptr<TensorCore> correct_ptr = mInputTensorCoreTbl[1].lock())
					{
						TensorCore& inference = *inference_ptr;
						const TensorCore& correct = *correct_ptr;

						if (inference.requiresGrad())/*���z�s�v�ȏ󋵂Ȃ�t�`�����X�L�b�v�ł���*/
						{
							if (m_on_cuda)
							{
								auto inference_gpu_address = inference.getGpuDataAddress();
								auto inference_gpu_grad_address = inference.getGpuGradDataAddress();
								auto correct_gpu_address = correct.getGpuDataAddress();

								dim3 block(32);
								dim3 grid((m_batch_size + block.x - 1) / block.x);
#ifdef TIME_DEBUG
								std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
								backward_gpu_impl << <grid, block >> > (
									inference_gpu_grad_address,
									inference_gpu_address,
									correct_gpu_address,
									m_batch_size,
									m_label_num);
								CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
								f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
								std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_gpu_impl");
								debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
							}
							else
							{
#ifdef TIME_DEBUG
								std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
								backward_cpu_impl(inference, correct);
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
						std::cout << "correctData resource Error@CrossEntropyWithSMCore::backward" << std::endl;
						exit(1);
					}
				}
				else
				{
					std::cout << "infereceData resource Error@CrossEntropyWithSMCore::backward" << std::endl;
					exit(1);
				}
			}


			void CrossEntropyWithSMCore::forward_cpu_impl(const TensorCore& inference, const TensorCore& correct)
			{
				DataType result = 0.0f;

				for (u32 N = 0, end = m_batch_size; N < end; N++)
				{
					DataType max = inference(N, 0);
					for (u32 i = 0; i < m_label_num; i++)
					{
						DataType cand = inference(N, i);
						if (max < cand)
						{
							max = cand;
						}
					}

					DataType sum = 0.0f;
					for (u32 i = 0; i < m_label_num; i++)
					{
						DataType value = inference(N, i) - max;
						sum += std::exp(value);
					}

					u32 correct_index = static_cast<u32>(correct(N));
					if (correct_index >= m_label_num)
					{
						assert(0);
					}
					DataType value = std::exp(inference(N, correct_index) - max) / sum;
					DataType loss = -std::log(value + 1e-7);
					result += loss;
					mLossPerBatch[N] = loss;//����͐����K�v�Ȃ��BGPU�ƃp�������ɂ����������ׁB�����Ă�OK�B
				}

				mOutput[0] = result / (m_batch_size);
			}

			void  CrossEntropyWithSMCore::backward_cpu_impl(TensorCore& inference, const TensorCore& correct)
			{
				for (u32 N = 0; N < m_batch_size; N++)
				{
					DataType max = inference(N, 0);
					for (u32 i = 0; i < m_label_num; i++)
					{
						DataType cand = inference(N, i);
						if (max < cand)
						{
							max = cand;
						}
					}

					DataType sum = 0.0f;
					for (u32 i = 0; i < m_label_num; i++)
					{
						DataType value = inference(N, i) - max;
						sum += exp(value);
					}

					const u32 correct_label = static_cast<u32>(correct(N));

					for (u32 I = 0; I < m_label_num; I++)
					{
						inference.d(N, I) = exp(inference(N, I) - max) / sum - (correct_label == I ? 1 : 0);
					}
				}
			}

		}
	}
}