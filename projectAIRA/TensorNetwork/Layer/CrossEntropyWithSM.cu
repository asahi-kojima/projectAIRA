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
		DataType* output,
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
	, mOutput(*m_output_tensorcore_tbl[0])
	, mLossPerBatch(false)
{
}



LayerSkeleton::iotype CrossEntropyWithSMCore::forward(const LayerSkeleton::iotype& input_tensors)
{


	const auto& inference = *getTensorCoreFrom(input_tensors[0]);
	const auto& correct = *getTensorCoreFrom(input_tensors[1]);


	//入力間での無矛盾性のチェック
	{
		if (inference.mBatchSize != correct.mBatchSize)
		{
			std::cout << "batchSize is not consistent." << std::endl;
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
		m_batch_size = inference.mBatchSize;
		m_label_num = inference.mCHW;

		//出力テンソルの形状変更
		bool isInit = mOutput.reshapeAs(1, inference.m_on_cuda);
		if (isInit)
		{
			mOutput.d(0) = 1;
		}

		//途中計算に必要なバッチ損失の形状変更
		mLossPerBatch.reshapeAs(m_batch_size, 1, inference.m_on_cuda);
	}





	if (m_on_cuda)
	{
		auto inference_gpu_address		= inference._m_gpu_data_address;
		auto correct_gpu_address		= correct._m_gpu_data_address;
		auto lossPerBatch_gpu_address	= mLossPerBatch._m_gpu_data_address;
		auto output_gpu_address			= mOutput._m_gpu_data_address;

		{
			dim3 block(32);
			dim3 grid((m_batch_size + block.x - 1) / block.x);
			forward_gpu_impl_pre << <grid, block >> > (lossPerBatch_gpu_address, inference_gpu_address, correct_gpu_address, m_batch_size, m_label_num);
			CUDA_SYNCHRONIZE_DEBUG;
		}
		{
			dim3 block(1);
			dim3 grid(1);
			forward_gpu_impl_sum << <grid, block >> > (output_gpu_address, lossPerBatch_gpu_address, m_batch_size);
			CUDA_SYNCHRONIZE_DEBUG;
		}

	}
	else
	{
		forward_cpu_impl(inference, correct);
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

			if (inference.m_grad_required)/*勾配不要な状況なら逆伝搬をスキップできる*/
			{
				if (m_on_cuda)
				{
					auto inference_gpu_address = inference._m_gpu_data_address;
					auto inference_gpu_grad_address = inference._m_gpu_grad_data_address;
					auto correct_gpu_address = correct._m_gpu_data_address;

					dim3 block(32);
					dim3 grid((m_batch_size + block.x - 1) / block.x);
					backward_gpu_impl << <grid, block >> > (
						inference_gpu_grad_address,
						inference_gpu_address,
						correct_gpu_address,
						m_batch_size,
						m_label_num);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					backward_cpu_impl(inference, correct);
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
		mLossPerBatch[N] = loss;//これは正直必要ない。GPUとパラレルにしたかった為。消してもOK。
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