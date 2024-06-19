#include "Split.h"
#include "Layer.h"

namespace
{
	__global__ void forward_impl_gpu(DataType* output0, DataType* output1, const DataType* input, u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		const DataType inputValue = input[xid];
		output0[xid] = inputValue;
		output1[xid] = inputValue;
	}


	__global__ void backward_impl_gpu(DataType* input_grad, const DataType* output0_grad, const DataType* output1_grad, u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		input_grad[i] = output0_grad[i] + output1_grad[i];
	}
}

using namespace aoba::nn::layer;


Layer Split()
{
	Layer add_layer = gen<SplitCore>("Split");
	return add_layer;
}


SplitCore::SplitCore()
	: BaseLayer(1, 2, 2)
	, m_data_size(0)
	, mOutput0(*m_output_tensorcore_tbl[0])
	, mOutput1(*m_output_tensorcore_tbl[1])
{
}



BaseLayer::iotype SplitCore::forward(const BaseLayer::iotype& input_tensors)
{
	if (!m_init_finish)
	{
		initialize();
	}

	const auto& input = *getTensorCoreFrom(input_tensors[0]);

	//出力テンソルと訓練パラメータの形状確認＆対応
	{
		m_data_size = input.getDataSize();

		//出力テンソルの形状変更
		mOutput0.reshapeAs(input, input.isOnCuda());
		mOutput1.reshapeAs(input, input.isOnCuda());
	}


	if (m_on_cuda)
	{
		auto input_gpu_address = input.getGpuDataAddress();
		auto output0_gpu_address = mOutput0.getGpuDataAddress();
		auto output1_gpu_address = mOutput1.getGpuDataAddress();

		dim3 block(32);
		dim3 grid((m_data_size + block.x - 1) / block.x);
		forward_impl_gpu << <grid, block >> > (output0_gpu_address, output1_gpu_address, input_gpu_address, m_data_size);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		forward_cpu_impl(input);
	}


	return iotype{ Tensor(m_output_tensorcore_tbl[0]), Tensor(m_output_tensorcore_tbl[1]) };
}



void SplitCore::backward()
{
	if (const std::shared_ptr<TensorCore>& input_tensorcore = mInputTensorCoreTbl[0].lock())
	{
		TensorCore& input = *input_tensorcore;

		if (input.requiresGrad())
		{
			if (m_on_cuda)
			{
				const auto output0_gpu_grad_address = mOutput0.getGpuGradDataAddress();
				const auto output1_gpu_grad_address = mOutput1.getGpuGradDataAddress();
				auto input_gpu_grad_address = input.getGpuGradDataAddress();

				dim3 block(32);
				dim3 grid((m_data_size + block.x - 1) / block.x);
				backward_impl_gpu << <grid, block >> > (input_gpu_grad_address, output0_gpu_grad_address, output1_gpu_grad_address, m_data_size);
				CUDA_SYNCHRONIZE_DEBUG;
			}
			else
			{
				backward_cpu_impl(input);
			}
		}
	}
	else
	{
		std::cout << "Resource Error : Input @ReLUCore::backward" << std::endl;
		exit(1);
	}
}

void SplitCore::forward_cpu_impl(const TensorCore& input)
{
	for (u32 i = 0; i < m_data_size; i++)
	{
		const DataType inputValue = input[i];
		mOutput0[i] = inputValue;
		mOutput1[i] = inputValue;
	}
}

void SplitCore::backward_cpu_impl(TensorCore& input)
{
	for (u32 i = 0; i < m_data_size; i++)
	{
		input.d(i) = mOutput0.d(i) + mOutput1.d(i);
	}
}