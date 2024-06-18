#include "Split.h"
#include "nnLayer.h"
namespace
{
	void split_forward_impl_cpu(DataType* input_ptr, DataType* output0_ptr, DataType* output1_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			output0_ptr[i] = input_ptr[i];
			output1_ptr[i] = input_ptr[i];
		}
	}

	__global__ void split_forward_impl_gpu(DataType* input_ptr, DataType* output0_ptr, DataType* output1_ptr, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output0_ptr[xid] = input_ptr[xid];
		output1_ptr[xid] = input_ptr[xid];
	}



	void split_backward_impl_cpu(DataType* d_output0_ptr, DataType* d_output1_ptr, DataType* d_input_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			d_input_ptr[i] = d_output0_ptr[i] + d_output1_ptr[i];
		}
	}

	__global__ void split_backward_impl_gpu(DataType* d_output0_ptr, DataType* d_output1_ptr, DataType* d_input_ptr, u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		d_input_ptr[i] = d_output0_ptr[i] + d_output1_ptr[i];
	}
}

using namespace aoba::nn::layer;
using SplitCore = Layer::SplitCore;
using LayerSkeleton = Layer::LayerSkeleton;


Layer::nnLayer Split()
{
	Layer::nnLayer add_layer = gen<SplitCore>("Add");
	return add_layer;
}


SplitCore::SplitCore()
	: LayerSkeleton(1, 2, 2)
{
}



LayerSkeleton::iotype SplitCore::forward(const LayerSkeleton::iotype& input_tensors)
{
	const auto& input_tensorcore = *getTensorCoreFrom(input_tensors[0]);


	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
	if (!m_init_finish)
	{
		auto& child_tensorcore0 = m_output_tensorcore_tbl[0];
		auto& child_tensorcore1 = m_output_tensorcore_tbl[1];

		//genDownStreamTensor(0, std::make_shared<TensorCore>(input_tensorcore, true));
		//genDownStreamTensor(1, std::make_shared<TensorCore>(input_tensorcore, true));

		if (input_tensorcore.m_on_cuda)
		{
			m_on_cuda = true;
			child_tensorcore0->to_cuda();
			child_tensorcore1->to_cuda();
		}
		m_init_finish = true;
	}

	const auto& child_tensorcore0 = *m_output_tensorcore_tbl[0];
	const auto& child_tensorcore1 = *m_output_tensorcore_tbl[1];

	auto dataSize_input = input_tensorcore.mDataSize;
	auto dataSize_output0 = child_tensorcore0.mDataSize;
	auto dataSize_output1 = child_tensorcore1.mDataSize;
	if (dataSize_input != dataSize_output0 || dataSize_input != dataSize_output1)
	{
		std::cout << "Input tensor size between Input & Output0 & Output1 is not match." << std::endl;
		exit(1);
	}



	std::cout << "Split forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
	if (m_on_cuda)
	{
		auto input_address = input_tensorcore._m_gpu_data_address;
		auto output0_address = child_tensorcore0._m_gpu_data_address;
		auto output1_address = child_tensorcore1._m_gpu_data_address;

		dim3 block(256);
		dim3 grid((dataSize_input + block.x - 1) / block.x);
		split_forward_impl_gpu << <grid, block >> > (input_address, output0_address, output1_address, dataSize_input);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		auto input_address = input_tensorcore._m_cpu_data_address;
		auto output0_address = child_tensorcore0._m_cpu_data_address;
		auto output1_address = child_tensorcore1._m_cpu_data_address;
		split_forward_impl_cpu(input_address, output0_address, output1_address, dataSize_input);
	}


	return iotype{ Tensor(m_output_tensorcore_tbl[0]), Tensor(m_output_tensorcore_tbl[1]) };
}



void SplitCore::backward()
{
	std::cout << "Add backward" << std::endl;
	if (std::shared_ptr<TensorCore> input_tensor_core = mInputTensorCoreTbl[0].lock())
	{
		bool need_grad = input_tensor_core->m_grad_required;

		if (need_grad)
		{
			auto dataSize = input_tensor_core->mDataSize;
			if (m_on_cuda)
			{
				auto output0_address = m_output_tensorcore_tbl[0]->_m_gpu_grad_data_address;
				auto output1_address = m_output_tensorcore_tbl[1]->_m_gpu_grad_data_address;
				auto input_address = input_tensor_core->_m_gpu_grad_data_address;

				dim3 block(256);
				dim3 grid((dataSize + block.x - 1) / block.x);
				split_backward_impl_gpu << <grid, block >> > (output0_address, output1_address, input_address, dataSize);
				CUDA_SYNCHRONIZE_DEBUG;
			}
			else
			{
				auto output0_address = m_output_tensorcore_tbl[0]->_m_cpu_grad_data_address;
				auto output1_address = m_output_tensorcore_tbl[1]->_m_cpu_grad_data_address;
				auto input_address = input_tensor_core->_m_cpu_grad_data_address;
				split_backward_impl_cpu(output0_address, output1_address, input_address, dataSize);
			}
		}
	}
	else
	{
		std::cout << "Resource Error : Input @ReLUCore::backward" << std::endl;
		exit(1);
	}
}

