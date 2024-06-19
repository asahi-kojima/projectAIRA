#include "Add.h"

namespace
{
	__global__ void forward_gpu_impl(
		DataType* output_ptr,
		const DataType* input0_ptr,
		const DataType* input1_ptr,
		const u32 data_size)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		if (xid >= data_size)
		{
			return;
		}

		output_ptr[xid] = input0_ptr[xid] + input1_ptr[xid];
	}



	__global__ void backward_gpu_impl_both(
		DataType* inputL_grad,
		DataType* inputR_grad,
		const DataType* output_grad,
		const u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		const DataType gradValue = output_grad[i];
		inputL_grad[i] = gradValue;
		inputR_grad[i] = gradValue;
	}

	__global__ void backward_gpu_impl_oneside(
		DataType* input_grad,
		const DataType* output_grad,
		const u32 data_size)
	{
		u32 i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= data_size)
		{
			return;
		}

		input_grad[i] = output_grad[i];
	}

}

using namespace aoba::nn::layer;


Layer Add()
{
	Layer add_layer = aoba::nn::layer::gen<AddCore>("Add");
	return add_layer;
}


AddCore::AddCore()
	: BaseLayer(2, 1, 1)
	, m_data_size(0)
	, mOutput(*m_output_tensorcore_tbl[0])
{
}



BaseLayer::iotype AddCore::forward(const BaseLayer::iotype& input_tensors)
{
	if (!m_init_finish)
	{
		initialize();
	}


	const auto& inputL = *getTensorCoreFrom(input_tensors[0]);
	const auto& inputR = *getTensorCoreFrom(input_tensors[1]);

	auto dataSize_L = inputL.getDataSize();
	auto dataSize_R = inputR.getDataSize();


	//形状は任意でいいが、要素数が一致していないと演算が出来ない。
	if (dataSize_L != dataSize_R)
	{
		std::cout << "Input tensor size between LHS & RHS is not equal@AddCore::forward" << std::endl;
		exit(1);
	}

	{
		m_data_size = dataSize_L;

		mOutput.reshapeAs(inputL, inputL.isOnCuda());
	}

	if (m_on_cuda)
	{
		auto inputL_gpu_address = inputL.getGpuDataAddress();
		auto inputR_gpu_address = inputR.getGpuDataAddress();
		auto output_gpu_address = mOutput.getGpuDataAddress();

		dim3 block(256);
		dim3 grid((m_data_size + block.x - 1) / block.x);
		forward_gpu_impl << <grid, block >> > (
			output_gpu_address,
			inputL_gpu_address,
			inputR_gpu_address,
			m_data_size);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	else
	{
		forward_cpu_impl(inputL, inputR);
	}

	return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
}



void AddCore::backward()
{
	if (const std::shared_ptr<TensorCore>& inputL_tensorcore_ptr = mInputTensorCoreTbl[0].lock())
	{
		if (const std::shared_ptr<TensorCore>& inputR_tensorcore_ptr = mInputTensorCoreTbl[1].lock())
		{
			TensorCore& inputL = *inputL_tensorcore_ptr;
			TensorCore& inputR = *inputR_tensorcore_ptr;


			bool inputL_requires_grad = inputL.requiresGrad();
			bool inputR_requires_grad = inputR.requiresGrad();
			if (!inputL_requires_grad && !inputR_requires_grad)/*どちらも勾配不要な状況なら逆伝搬をスキップできる*/
			{
				if (m_on_cuda)
				{
					auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();
					auto inputL_gpu_grad_address = inputL.getGpuGradDataAddress();
					auto inputR_gpu_grad_address = inputR.getGpuGradDataAddress();

					dim3 block(32);
					dim3 grid((m_data_size + block.x - 1) / block.x);

					if (inputL_requires_grad && inputR_requires_grad)
					{
						backward_gpu_impl_both<<<grid, block>>>(inputL_gpu_grad_address, inputR_gpu_grad_address, output_gpu_grad_address, m_data_size);
					}
					else if (inputL_requires_grad && !inputR_requires_grad)
					{
						backward_gpu_impl_oneside << <grid, block >> > (inputL_gpu_grad_address, output_gpu_grad_address, m_data_size);
					}
					else if (!inputL_requires_grad && inputR_requires_grad)
					{
						backward_gpu_impl_oneside << <grid, block >> > (inputR_gpu_grad_address, output_gpu_grad_address, m_data_size);
					}
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					backward_cpu_impl(inputL, inputR);
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


void AddCore::forward_cpu_impl(const TensorCore& inputL, const TensorCore& inputR)
{
	for (u32 i = 0; i < m_data_size; i++)
	{
		mOutput[i] = inputL[i] + inputR[i];
	}
}

void AddCore::backward_cpu_impl(TensorCore& inputL, TensorCore& inputR)
{
	bool inputL_need_grad = inputL.requiresGrad();
	bool inputR_need_grad = inputR.requiresGrad();

	if (inputL_need_grad && inputR_need_grad)
	{
		for (u32 i = 0; i < m_data_size; i++)
		{
			const DataType gradValue = mOutput.d(i);
			inputL.d(i) =gradValue;
			inputR.d(i) =gradValue;
		}
	}
	else if (inputL_need_grad && !inputR_need_grad)
	{
		for (u32 i = 0; i < m_data_size; i++)
		{
			inputL.d(i) = mOutput.d(i);
		}
	}
	else if (!inputL_need_grad && inputR_need_grad)
	{
		for (u32 i = 0; i < m_data_size; i++)
		{
			inputR.d(i) = mOutput.d(i);
		}
	}
	else
	{
	}
}