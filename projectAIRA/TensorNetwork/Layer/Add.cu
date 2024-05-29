#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Add.h"
#include "Tensor/TensorCore.h"

namespace
{
	void add_impl_cpu(DataType* output_ptr, DataType* input0_ptr, DataType* input1_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			output_ptr[i] = input0_ptr[i] + input1_ptr[i];
		}
	}

	__global__ void add_imple_gpu(DataType* output_ptr, DataType* input0_ptr, DataType* input1_ptr, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output_ptr[xid] = input0_ptr[xid] + input1_ptr[xid];
	}

}

LayerCore::iotype AddCore::forward(const LayerCore::iotype& input_tensors)
{
	std::cout << "Add forward " << (m_use_gpu ? "On GPU" : "on CPU") << std::endl;
	auto dataSize_lhs = Accessor2TensorCore::getDataSize(input_tensors[0]);
	auto dataSize_rhs = Accessor2TensorCore::getDataSize(input_tensors[1]);

	if (dataSize_lhs != dataSize_rhs)
	{
		std::cout << "Input tensor size between LHS & RHS is not equal@AddCore::forward" << std::endl;
		exit(1);
	}
	if (!m_init_finish)
	{
		std::vector<u32> shape = Accessor2TensorCore::getTensorShape(input_tensors[0]);
		auto& child_tensorcore = m_child_tensorcore_tbl[0];
		child_tensorcore = std::make_shared<TensorCore>(true, shape);
		child_tensorcore->regist_parent_layercore(shared_from_this());
		m_init_finish = true;
	}

	auto dataSize = Accessor2TensorCore::getDataSize(m_child_tensorcore_tbl[0]);
	if (dataSize_lhs != dataSize_rhs || dataSize != dataSize_lhs)
	{
		std::cout << "Input tensor size between LHS & RHS & Output is not match." << std::endl;
		exit(1);
	}



	if (m_use_gpu)
	{
		auto output_address = Accessor2TensorCore::getAddressOnGpuFrom(m_child_tensorcore_tbl[0]);
		auto input_address0 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensors[0]);
		auto input_address1 = Accessor2TensorCore::getAddressOnGpuFrom(input_tensors[1]);

		dim3 block(256);
		dim3 grid((dataSize + block.x - 1) / block.x);
		add_imple_gpu << <grid, block >> > (output_address, input_address0, input_address1, dataSize);
	}
	else
	{
		auto output_address = Accessor2TensorCore::getAddressOnCpuFrom(m_child_tensorcore_tbl[0]);
		auto input_address0 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensors[0]);
		auto input_address1 = Accessor2TensorCore::getAddressOnCpuFrom(input_tensors[1]);
		add_impl_cpu(output_address, input_address0, input_address1, dataSize);
	}

	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
}


