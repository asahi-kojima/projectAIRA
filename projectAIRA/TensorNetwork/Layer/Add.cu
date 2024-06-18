#include "Add.h"

namespace
{
	void add_forward_cpu_impl(DataType* input0_ptr, DataType* input1_ptr, DataType* output_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			output_ptr[i] = input0_ptr[i] + input1_ptr[i];
		}
	}

	__global__ void add_forward_gpu_impl(DataType* input0_ptr, DataType* input1_ptr, DataType* output_ptr, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output_ptr[xid] = input0_ptr[xid] + input1_ptr[xid];
	}



	void add_backward_cpu_impl(DataType* d_input0_ptr, bool need_grad0, DataType* d_input1_ptr, bool need_grad1, DataType* d_output_ptr, u32 data_size)
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

	__global__ void add_backward_gpu_impl(DataType* d_input0_ptr, bool need_grad0, DataType* d_input1_ptr, bool need_grad1, DataType* d_output_ptr, u32 data_size)
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


Layer Add()
{
	Layer add_layer = aoba::nn::layer::gen<AddCore>("Add");
	return add_layer;
}


AddCore::AddCore()
	: LayerBase(2, 1, 1)
{
}



LayerBase::iotype AddCore::forward(const LayerBase::iotype& input_tensors)
{
	const auto& input_tensorcore0 = *getTensorCoreFrom(input_tensors[0]);
	const auto& input_tensorcore1 = *getTensorCoreFrom(input_tensors[1]);


	auto dataSize_lhs = input_tensorcore0.mDataSize;
	auto dataSize_rhs = input_tensorcore1.mDataSize;

	//形状は任意でいいが、要素数が一致していないと演算が出来ない。
	if (dataSize_lhs != dataSize_rhs)
	{
		std::cout << "Input tensor size between LHS & RHS is not equal@AddCore::forward" << std::endl;
		exit(1);
	}

	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
	if (!m_init_finish)
	{
		auto& child_tensorcore = m_output_tensorcore_tbl[0];
		genDownStreamTensor(0);


		if (input_tensorcore0.m_on_cuda)
		{
			m_on_cuda = true;
			child_tensorcore->to_cuda();
		}
		m_init_finish = true;
	}

	const auto& child_tensorcore = *m_output_tensorcore_tbl[0];

	auto dataSize = child_tensorcore.mDataSize;
	if (dataSize_lhs != dataSize_rhs || dataSize != dataSize_lhs)
	{
		std::cout << "Input tensor size between LHS & RHS & Output is not match." << std::endl;
		exit(1);
	}



	std::cout << "Add forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
	if (m_on_cuda)
	{
		auto input_address0 = input_tensorcore0._m_gpu_data_address;
		auto input_address1 = input_tensorcore1._m_gpu_data_address;
		auto output_address = child_tensorcore._m_gpu_data_address;

		dim3 block(256);
		dim3 grid((dataSize + block.x - 1) / block.x);
		add_forward_gpu_impl << <grid, block >> > (input_address0, input_address1, output_address, dataSize);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		auto input_address0 = input_tensorcore0._m_cpu_data_address;
		auto input_address1 = input_tensorcore1._m_cpu_data_address;
		auto output_address = child_tensorcore._m_cpu_data_address;
		add_forward_cpu_impl(input_address0, input_address1, output_address, dataSize);
	}

	return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
}



void AddCore::backward()
{
	std::cout << "Add backward" << std::endl;
	if (std::shared_ptr<TensorCore> input_tensor_core0 = mInputTensorCoreTbl[0].lock())
	{
		if (std::shared_ptr<TensorCore> input_tensor_core1 = mInputTensorCoreTbl[1].lock())
		{
			bool need_grad0 = input_tensor_core0->m_grad_required;
			bool need_grad1 = input_tensor_core1->m_grad_required;
			if (need_grad0 || need_grad1)/*どちらも勾配不要な状況なら逆伝搬をスキップできる*/
			{

				auto dataSize = m_output_tensorcore_tbl[0]->mDataSize;
				if (m_on_cuda)
				{
					auto output_address = m_output_tensorcore_tbl[0]->_m_gpu_grad_data_address;
					auto input_address0 = input_tensor_core0->_m_gpu_grad_data_address;
					auto input_address1 = input_tensor_core1->_m_gpu_grad_data_address;

					dim3 block(256);
					dim3 grid((dataSize + block.x - 1) / block.x);
					add_backward_gpu_impl << <grid, block >> > (input_address0, need_grad0, input_address1, need_grad1, output_address, dataSize);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					auto output_address = m_output_tensorcore_tbl[0]->_m_cpu_grad_data_address;
					auto input_address0 = input_tensor_core0->_m_cpu_grad_data_address;
					auto input_address1 = input_tensor_core1->_m_cpu_grad_data_address;
					add_backward_cpu_impl(input_address0, need_grad0, input_address1, need_grad1, output_address, dataSize);
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

