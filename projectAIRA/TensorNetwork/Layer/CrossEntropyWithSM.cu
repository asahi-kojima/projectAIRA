#include "CrossEntropyWithSM.h"
#include "nnLayer.h"

namespace
{
	void relu_forward_cpu_impl(DataType* input0_ptr, DataType* input1_ptr, DataType* output_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			output_ptr[i] = input0_ptr[i] + input1_ptr[i];
		}
	}

	__global__ void relu_forward_gpu_impl(DataType* input0_ptr, DataType* input1_ptr, DataType* output_ptr, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output_ptr[xid] = input0_ptr[xid] + input1_ptr[xid];
	}



	void relu_backward_cpu_impl(DataType* d_input0_ptr, bool need_grad0, DataType* d_input1_ptr, bool need_grad1, DataType* d_output_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			if (need_grad0)
			{
				d_input0_ptr[i] = d_output_ptr[i];
			}
			if (need_grad1)
			{
				d_input1_ptr[i] = d_output_ptr[i];
			}
		}
	}

	__global__ void relu_backward_gpu_impl(DataType* d_input0_ptr, bool need_grad0, DataType* d_input1_ptr, bool need_grad1, DataType* d_output_ptr, u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		if (need_grad0)
		{
			d_input0_ptr[i] = d_output_ptr[i];
		}
		if (need_grad1)
		{
			d_input1_ptr[i] = d_output_ptr[i];
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
{
}



LayerSkeleton::iotype CrossEntropyWithSMCore::forward(const LayerSkeleton::iotype& input_tensors)
{
	const auto& inference = *getTensorCoreFrom(input_tensors[0]);
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
		child_tensorcore->regist_parent_layercore(shared_from_this());

		if (inference._m_on_cuda)
		{
			m_on_cuda = true;
			child_tensorcore->to_cuda("");
		}


		m_init_finish = true;
	}


	const auto& loss = *m_child_tensorcore_tbl[0];

	if (batchSize_lhs != m_batch_size)
	{
		std::cout << "Input'Batch size  & Initialized Batch size don't match@CrossEntropyWithSMCore::forward" << std::endl;
		exit(1);
	}



	//std::cout << "CrossEntropyWithSM forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
	if (m_on_cuda)
	{
		//auto input_address0 = input_tensorcore0._m_gpu_data_address;
		//auto input_address1 = input_tensorcore1._m_gpu_data_address;
		//auto output_address = child_tensorcore._m_gpu_data_address;

		//dim3 block(256);
		//dim3 grid((dataSize + block.x - 1) / block.x);
		//relu_forward_gpu_impl << <grid, block >> > (input_address0, input_address1, output_address, dataSize);
		//CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		crossEntropyWithSM_forward_cpu_impl(input_tensors);
	}

	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
}



void CrossEntropyWithSMCore::backward()
{
	//std::cout << "CrossEntropyWithSM backward" << std::endl;
	if (std::shared_ptr<TensorCore> input_tensor_core0 = mInputTensorCoreTbl[0].lock())
	{
		if (std::shared_ptr<TensorCore> input_tensor_core1 = mInputTensorCoreTbl[1].lock())
		{
			bool need_grad0 = input_tensor_core0->_m_need_grad;
			bool need_grad1 = input_tensor_core1->_m_need_grad;
			if (need_grad0 || need_grad1)/*どちらも勾配不要な状況なら逆伝搬をスキップできる*/
			{

				auto dataSize = m_child_tensorcore_tbl[0]->mDataSize;
				if (m_on_cuda)
				{
					auto output_address = m_child_tensorcore_tbl[0]->_m_gpu_grad_data_address;
					auto input_address0 = input_tensor_core0->_m_gpu_grad_data_address;
					auto input_address1 = input_tensor_core1->_m_gpu_grad_data_address;

					dim3 block(256);
					dim3 grid((dataSize + block.x - 1) / block.x);
					relu_backward_gpu_impl << <grid, block >> > (input_address0, need_grad0, input_address1, need_grad1, output_address, dataSize);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					auto output_address = m_child_tensorcore_tbl[0]->_m_cpu_grad_data_address;
					auto input_address0 = input_tensor_core0->_m_cpu_grad_data_address;
					auto input_address1 = input_tensor_core1->_m_cpu_grad_data_address;
					relu_backward_cpu_impl(input_address0, need_grad0, input_address1, need_grad1, output_address, dataSize);
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


void CrossEntropyWithSMCore::crossEntropyWithSM_forward_cpu_impl(const LayerSkeleton::iotype& input_tensors)
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
			sum += exp(value);
		}

		u32 correct_index = static_cast<u32>(correct(N));
		DataType loss = -log(exp(inference(N, correct_index) - max) / sum + 1e-7);
		result += loss;
	}

	output(0) = result / (m_batch_size * m_label_num);
}

