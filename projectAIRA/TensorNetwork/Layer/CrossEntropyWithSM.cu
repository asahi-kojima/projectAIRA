#include "CrossEntropyWithSM.h"
#include "nnLayer.h"

namespace
{
	__global__ void forward_gpu_impl_pre(
		DataType* lossPerBatch, 
		DataType* inference,
		DataType* correct, 
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
		DataType* output , 
		DataType* lossPerBatch, 
		const u32 batchSize)
	{
		u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		DataType sum = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
			sum += lossPerBatch[N];
		}
		output[0] = sum / batchSize;
		//printf("%d | %f\n",N,  sum / batchSize);
	}

	__global__ void backward_gpu_impl(
		DataType* d_inference, 
		DataType* inference, 
		DataType* correct, 
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

using namespace aoba::nn::layer;
using CrossEntropyWithSMCore = Layer::CrossEntropyWithSMCore;
using LayerSkeleton = Layer::LayerSkeleton;

Layer::nnLayer aoba::nn::layer::CrossEntropyWithSM()
{
	Layer::nnLayer add_layer = gen<CrossEntropyWithSMCore>("CrossEntropyWithSM");
	return add_layer;
}


CrossEntropyWithSMCore::CrossEntropyWithSMCore()
	: LayerSkeleton(2, 1, 1)
	, m_batch_size(0)
	, m_label_num(0)
	, mLossPerBatch()
{
}



LayerSkeleton::iotype CrossEntropyWithSMCore::forward(const LayerSkeleton::iotype& input_tensors)
{
	auto& inference = *getTensorCoreFrom(input_tensors[0]);
	const auto& correct = *getTensorCoreFrom(input_tensors[1]);


	const auto batchSize_lhs = inference.mBatchSize;
	const auto dataSize_lhs = inference.mCHW;
	const auto batchSize_rhs = correct.mBatchSize;
	const auto dataSize_rhs = correct.mCHW;

	//バッチ数が合っていないと計算できない。
	if (batchSize_lhs != batchSize_rhs)
	{
		std::cout << "Batch size between LHS & RHS tensor is not equal@CrossEntropyWithSMCore::forward" << std::endl;
		exit(1);
	}

	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
	if (!m_init_finish)
	{
		m_batch_size = batchSize_lhs;
		m_label_num = dataSize_lhs;

		auto& child_tensorcore = m_child_tensorcore_tbl[0];
		child_tensorcore = std::make_shared<TensorCore>(1, true);
		child_tensorcore->_m_location_in_upstream_layer = 0;
		child_tensorcore->regist_parent_layercore(shared_from_this());
		child_tensorcore->d(0) = 1;

		mLossPerBatch = Tensor(m_batch_size, 1, false);

		if (inference._m_on_cuda)
		{
			m_on_cuda = true;
			child_tensorcore->to_cuda("");

			mLossPerBatch.to_cuda(true);
		}


		m_init_finish = true;
	}


	const auto& loss = *m_child_tensorcore_tbl[0];

	if (batchSize_lhs != m_batch_size)
	{
		std::cout << "Input'Batch size  & Initialized Batch size don't match@CrossEntropyWithSMCore::forward" << std::endl;
		exit(1);
	}



	if (m_on_cuda)
	{
		auto inference_address = inference._m_gpu_data_address;
		auto correct_address = correct._m_gpu_data_address;
		auto lossPerBatch_address = mLossPerBatch.pTensorCore->_m_gpu_data_address;
		auto output_address = m_child_tensorcore_tbl[0]->_m_gpu_data_address;

		{
			dim3 block(32);
			dim3 grid((m_batch_size + block.x - 1) / block.x);
			forward_gpu_impl_pre << <grid, block >> > (lossPerBatch_address, inference_address, correct_address, m_batch_size, m_label_num);
			CUDA_SYNCHRONIZE_DEBUG;
		}
		{
			dim3 block(1);
			dim3 grid(1);
			forward_gpu_impl_sum << <grid, block >> > (output_address, lossPerBatch_address, m_batch_size);
			CUDA_SYNCHRONIZE_DEBUG;
		}
		
	}
	else
	{
		forward_cpu_impl(input_tensors);
	}

	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
}



void CrossEntropyWithSMCore::backward()
{
	//std::cout << "CrossEntropyWithSM backward" << std::endl;
	if (std::shared_ptr<TensorCore> inference = mInputTensorCoreTbl[0].lock())
	{
		if (std::shared_ptr<TensorCore> correct = mInputTensorCoreTbl[1].lock())
		{
			bool need_grad0 = inference->_m_need_grad;
			bool need_grad1 = correct->_m_need_grad;
			if (need_grad0 || need_grad1)/*どちらも勾配不要な状況なら逆伝搬をスキップできる*/
			{
				if (m_on_cuda)
				{
					auto inference_address = inference->_m_gpu_data_address;
					auto d_inference_address = inference->_m_gpu_grad_data_address;
					auto correct_address = correct->_m_gpu_data_address;

					dim3 block(32);
					dim3 grid((m_batch_size + block.x - 1) / block.x);
					backward_gpu_impl << <grid, block >> > (
						d_inference_address, 
						inference_address,
						correct_address,
						m_batch_size, 
						m_label_num);
					CUDA_SYNCHRONIZE_DEBUG;
					//inference->synchronize_from_GPU_to_CPU();
				}
				else
				{
					backward_cpu_impl(inference, correct);
				}
			}
		}
		else
		{
			std::cout << "Resource1 Error@ReLUCore::backward" << std::endl;
			exit(1);
		}
	}
	else
	{
		std::cout << "Resource0 Error@ReLUCore::backward" << std::endl;
		exit(1);
	}
}


void CrossEntropyWithSMCore::forward_cpu_impl(const LayerSkeleton::iotype& input_tensors)
{
	const auto& inference = *getTensorCoreFrom(input_tensors[0]);
	const auto& correct = *getTensorCoreFrom(input_tensors[1]);
	auto& output = *m_child_tensorcore_tbl[0];

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
		mLossPerBatch(N) = loss;//これは正直必要ない。GPUとパラレルにしたかった為。消してもOK。
	}

	output(0) = result / (m_batch_size);
}

void  CrossEntropyWithSMCore::backward_cpu_impl(const std::shared_ptr<TensorCore>& pInferenceData, const std::shared_ptr<TensorCore>& pCorrectData)
{
	auto& inferenceData = *pInferenceData;
	const auto& correctData = *pCorrectData;

	for (u32 N = 0; N < m_batch_size; N++)
	{
		DataType max = inferenceData(N, 0);
		for (u32 i = 0; i < m_label_num; i++)
		{
			DataType cand = inferenceData(N, i);
			if (max < cand)
			{
				max = cand;
			}
		}

		DataType sum = 0.0f;
		for (u32 i = 0; i < m_label_num; i++)
		{
			DataType value = inferenceData(N, i) - max;
			sum += exp(value);
		}

		const u32 correct_label = static_cast<u32>(correctData(N));

		for (u32 I = 0; I < m_label_num; I++)
		{
			inferenceData.d(N, I) = exp(inferenceData(N, I) - max) / sum - (correct_label == I ? 1 : 0);
		}
	}
}