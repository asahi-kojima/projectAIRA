#include "Add.h"

namespace
{
	void relu_forward_impl_cpu(DataType* output_ptr, DataType* input0_ptr, DataType* input1_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			output_ptr[i] = input0_ptr[i] + input1_ptr[i];
		}
	}

	__global__ void relu_forward_impl_gpu(DataType* output_ptr, DataType* input0_ptr, DataType* input1_ptr, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output_ptr[xid] = input0_ptr[xid] + input1_ptr[xid];
	}



	void relu_backward_impl_cpu(DataType* d_output_ptr, DataType* d_input0_ptr,bool need_grad0,  DataType* d_input1_ptr,bool need_grad1,  u32 data_size)
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

	__global__ void relu_backward_impl_gpu(DataType* d_output_ptr, DataType* d_input0_ptr, bool need_grad0, DataType* d_input1_ptr, bool need_grad1, u32 data_size)
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

Layer Add()
{
	Layer add_layer = gen<AddCore>("Add");
	return add_layer;
}


AddCore::AddCore()
	: LayerCore(2, 1, 1)
{
}



LayerCore::iotype AddCore::forward(const LayerCore::iotype& input_tensors)
{
	auto dataSize_lhs = Accessor2TensorCore::getDataSize(input_tensors[0]);
	auto dataSize_rhs = Accessor2TensorCore::getDataSize(input_tensors[1]);

	//形状は任意でいいが、要素数が一致していないと演算が出来ない。
	if (dataSize_lhs != dataSize_rhs)
	{
		std::cout << "Input tensor size between LHS & RHS is not equal@AddCore::forward" << std::endl;
		exit(1);
	}

	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
	if (!m_init_finish)
	{
		std::vector<u32> shape = Accessor2TensorCore::getTensorShape(input_tensors[0]);
		auto& child_tensorcore = m_child_tensorcore_tbl[0];
		child_tensorcore = std::make_shared<TensorCore>(true, shape);
		child_tensorcore->regist_parent_layercore(shared_from_this());

		if (Accessor2TensorCore::on_cuda(input_tensors[0]))
		{
			child_tensorcore->to_cuda("");

			//内部パラメータもCUDAに送る。
			m_on_cuda = true;
			for (u32 i = 0, end = m_parameter_tbl.size(); i < end; i++)
			{
				m_parameter_tbl[i]->to_cuda("");
			}
		}
		m_init_finish = true;
	}

	auto dataSize = Accessor2TensorCore::getDataSize(m_child_tensorcore_tbl[0]);
	if (dataSize_lhs != dataSize_rhs || dataSize != dataSize_lhs)
	{
		std::cout << "Input tensor size between LHS & RHS & Output is not match." << std::endl;
		exit(1);
	}



	std::cout << "Add forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
	if (m_on_cuda)
	{
		auto output_address = Accessor2TensorCore::getAddressOnGpuFrom(m_child_tensorcore_tbl[0]);
		auto input_address0 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensors[0]);
		auto input_address1 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensors[1]);

		dim3 block(256);
		dim3 grid((dataSize + block.x - 1) / block.x);
		relu_forward_impl_gpu << <grid, block >> > (output_address, input_address0, input_address1, dataSize);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		auto output_address = Accessor2TensorCore::getAddressOnCpuFrom(m_child_tensorcore_tbl[0]);
		auto input_address0 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensors[0]);
		auto input_address1 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensors[1]);
		relu_forward_impl_cpu(output_address, input_address0, input_address1, dataSize);
	}

	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
}



void AddCore::backward()
{
	std::cout << "Add backward" << std::endl;
	if (std::shared_ptr<TensorCore> input_tensor_core0 = mInputTensorCoreTbl[0].lock())
	{
		if (std::shared_ptr<TensorCore> input_tensor_core1 = mInputTensorCoreTbl[1].lock())
		{
			bool need_grad0 = Accessor2TensorCore::get_need_grad(input_tensor_core0);
			bool need_grad1 = Accessor2TensorCore::get_need_grad(input_tensor_core1);
			if (need_grad0 || need_grad1)
			{

				auto dataSize = Accessor2TensorCore::getDataSize(m_child_tensorcore_tbl[0]);
				if (m_on_cuda)
				{
					auto output_address = Accessor2TensorCore::getAddressOnGpuFrom(m_child_tensorcore_tbl[0]);
					auto input_address0 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensor_core0);
					auto input_address1 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensor_core1);

					dim3 block(256);
					dim3 grid((dataSize + block.x - 1) / block.x);
					relu_backward_impl_gpu << <grid, block >> > (output_address, input_address0, need_grad0, input_address1, need_grad1, dataSize);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					auto output_address = Accessor2TensorCore::getAddressOnCpuFrom(m_child_tensorcore_tbl[0]);
					auto input_address0 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensor_core0);
					auto input_address1 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensor_core1);
					relu_backward_impl_cpu(output_address, input_address0, need_grad0, input_address1, need_grad1, dataSize);
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
