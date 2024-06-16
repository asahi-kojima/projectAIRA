#include "ReLU.h"
#include "nnLayer.h"



namespace
{
	void relu_forward_impl_cpu(DataType* output_ptr, DataType* input_ptr, DataType* mask_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			const auto& input_value = input_ptr[i];
			auto& mask = mask_ptr[i];
			if (input_value > 0)
				mask = 1;
			else
				mask = 0;
			output_ptr[i] = input_value * mask;
		}
	}

	void relu_backward_impl_cpu(DataType* d_output_ptr, DataType* d_input_ptr, DataType* mask_ptr, u32 data_size)
	{
		for (u32 i = 0; i < data_size; i++)
		{
			d_input_ptr[i] = d_output_ptr[i] * mask_ptr[i];
		}
	}


	__global__ void relu_forward_impl_gpu(DataType* output_ptr, DataType* input_ptr, DataType* mask_ptr, u32 dataSize)
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


	__global__ void relu_backward_impl_gpu(DataType* doutput_ptr, DataType* dinput_ptr, DataType* mask_ptr, u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		doutput_ptr[i] = dinput_ptr[i] * mask_ptr[i];
	}
}

using namespace aoba::nn::layer;
using ReLUCore = Layer::ReLUCore;
using LayerSkeleton = Layer::LayerSkeleton;

Layer::nnLayer aoba::nn::layer::ReLU()
{
	Layer::nnLayer relu = gen<ReLUCore>("Add");
	return relu;
}

ReLUCore::ReLUCore()
	: LayerSkeleton(1, 1, 1, 1)
{
}


LayerSkeleton::iotype ReLUCore::forward(const LayerSkeleton::iotype& input_tensors)
{
	const auto& input_tensorcore = *getTensorCoreFrom(input_tensors[0]);

	auto dataSize_input = input_tensorcore.mDataSize;


	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
	if (!m_init_finish)
	{
		auto& child_tensorcore = m_output_tensorcore_tbl[0];
		genDownStreamTensor(0, std::make_shared<TensorCore>(input_tensorcore, true));

		auto& mask = m_parameter_tbl[0];
		mask = std::make_shared<TensorCore>(input_tensorcore, false);

		if (input_tensorcore._m_on_cuda)
		{
			m_on_cuda = true;

			child_tensorcore->to_cuda("");

			//内部パラメータもCUDAに送る。
			mask->to_cuda("");
		}
		m_init_finish = true;
	}

	auto& child_tensorcore = *m_output_tensorcore_tbl[0];
	const auto& mask = *m_parameter_tbl[0];
	auto dataSize = child_tensorcore.mDataSize;
	if (dataSize != dataSize_input)
	{
		std::cout << "Input tensor size between Input & Output is not match." << std::endl;
		exit(1);
	}


	if (m_on_cuda)
	{
		auto output_address = child_tensorcore._m_gpu_data_address;
		auto input_address = input_tensorcore._m_gpu_data_address;
		auto mask_address = mask._m_gpu_data_address;

		dim3 block(256);
		dim3 grid((dataSize + block.x - 1) / block.x);
		relu_forward_impl_gpu << <grid, block >> > (output_address, input_address, mask_address, dataSize);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		auto output_address = child_tensorcore._m_cpu_data_address;
		auto input_address = input_tensorcore._m_cpu_data_address;
		auto mask_address = mask._m_cpu_data_address;
		relu_forward_impl_cpu(output_address, input_address, mask_address, dataSize);
	}

	return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
}


void ReLUCore::backward()
{
	if (std::shared_ptr<TensorCore> input_tensor_core = mInputTensorCoreTbl[0].lock())
	{
		if (input_tensor_core->_m_need_grad)
		{
			const auto& output_tensorcore = *m_output_tensorcore_tbl[0];
			auto& input_tensorcore = *input_tensor_core;
			const auto& mask = *m_parameter_tbl[0];

			auto dataSize = output_tensorcore.mDataSize;
			if (m_on_cuda)
			{
				auto d_output_address = output_tensorcore._m_gpu_grad_data_address;
				auto d_input_address = input_tensorcore._m_gpu_grad_data_address;
				auto mask_address = mask._m_gpu_data_address;

				dim3 block(32);
				dim3 grid((dataSize + block.x - 1) / block.x);
				relu_backward_impl_gpu << <grid, block >> > (d_input_address, d_output_address, mask_address, dataSize);
				CUDA_SYNCHRONIZE_DEBUG;
				//input_tensorcore.synchronize_from_GPU_to_CPU();
			}
			else
			{
				auto output_grad_address = output_tensorcore._m_cpu_grad_data_address;
				auto input_grad_address = input_tensorcore._m_cpu_grad_data_address;
				auto mask_address = mask._m_cpu_data_address;
				relu_backward_impl_cpu(output_grad_address, input_grad_address, mask_address, dataSize);

			}
		}
	}
	else
	{
		std::cout << "Resource Error@ReLUCore::backward" << std::endl;
		exit(1);
	}
}

