
#include "Affine.h"
#include <random>

namespace
{


	__global__ void affine_forward_gpu_impl(
		f32* y, 
		f32* x, 
		f32* A,
		f32* b, 
		u32 batchSize, u32 outputSize, u32 inputSize)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
		if (xid >= outputSize || yid >= batchSize)
		{
			return;
		}
		u32 id = yid * outputSize + xid;
		f32 result = 0.0f;
		for (u32 i = 0; i < inputSize; i++)
		{
			result += A[xid * inputSize + i] * x[yid * inputSize + i];
		}
		y[id] = result + b[xid];
	}

	__global__ void affine_backward_gpu_impl_input(DataType* dOut, DataType* dIn, DataType* A, u32 batchSize, u32 outputSize, u32 inputSize)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;//input
		u32 yid = blockIdx.y * blockDim.y + threadIdx.y;//batch

		if (xid >= inputSize || yid >= batchSize)
		{
			return;
		}

		f32 result = 0.0f;
		for (u32 i = 0; i < outputSize; i++)
		{
#if INDEX_DEBUG
			if (i * inputSize + xid >= outputSize * inputSize)
			{
				assert(0);
			}
			if (yid * outputSize + i >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += A[i * inputSize + xid] * dIn[yid * outputSize + i];
		}
		dOut[yid * inputSize + xid] = result;
	}

	__global__ void affine_backward_gpu_impl_weight(DataType* dA, DataType* dout, DataType* input, u32 batchSize, u32 outputSize, u32 inputSize)
	{
		u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
		u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
		if (xid >= inputSize || yid >= outputSize)
		{
			return;
		}

		u32 id = yid * inputSize + xid;

		f32 result = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
#if INDEX_DEBUG
			if (N * inputSize + xid >= batchSize * inputSize)
			{
				assert(0);
			}
			if (N * outputSize + yid >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += dout[N * outputSize + yid] * input[N * inputSize + xid];
		}

		dA[id] = result;
	}

	__global__ void affine_backward_gpu_impl_bias(DataType* dBias, DataType* dout, u32 batchSize, u32 outputSize)
	{
		u32 id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= outputSize)
		{
			return;
		}
		f32 result = 0.0f;
		for (u32 N = 0; N < batchSize; N++)
		{
#if INDEX_DEBUG
			if ((N * outputSize + id) >= batchSize * outputSize)
			{
				assert(0);
			}
#endif
			result += dout[N * outputSize + id];
		}
#if INDEX_DEBUG
		if (id >= outputSize)
		{
			assert(0);
		}
#endif
		dBias[id] = result;
	}
}

using AffineCore = aoba::nn::layer::AffineCore;
using LayerCore = aoba::nn::layer::LayerCore;


Layer Affine(u32 output_size)
{
	Layer affine = aoba::nn::layer::gen<AffineCore>("Affine", output_size);
	return affine;
}




AffineCore::AffineCore(u32 output_size)
	:LayerCore(1, 1, 1, 2)
	, m_batch_size(0)
	, m_input_size(0)
	, m_output_size(output_size)
{
}

AffineCore::~AffineCore()
{}

LayerCore::iotype  AffineCore::forward(const LayerCore::iotype& input_tensors)
{
	const auto& input_tensorcore = *getTensorCoreFrom(input_tensors[0]);

	auto dataSize_input = input_tensorcore.mDataSize;

	if (!m_init_finish)
	{
		m_batch_size = input_tensorcore.mBatchSize;
		m_input_size = input_tensorcore.mCHW;

		auto& child_tensorcore = m_child_tensorcore_tbl[0];
		child_tensorcore = std::make_shared<TensorCore>(input_tensorcore.mBatchSize, m_output_size, true);
		child_tensorcore->regist_parent_layercore(shared_from_this());

		auto& weight = m_parameter_tbl[0];
		auto& bias = m_parameter_tbl[1];
		weight = std::make_shared<TensorCore>(m_output_size, input_tensorcore.mCHW, true);
		bias = std::make_shared<TensorCore>(m_output_size, true);

		{
			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());
			std::normal_distribution<> dist(0.0f, std::sqrt(2.0f / m_input_size));
			for (u32 i = 0; i < weight->mDataSize; i++)
			{
				weight->_m_cpu_data_address[i] = affineWeight * static_cast<DataType>(dist(engine));
			}
			for (u32 i = 0; i < bias->mDataSize; i++)
			{
				bias->_m_cpu_data_address[i] = 0.0f;
			}
		}

		if (input_tensorcore._m_on_cuda)
		{
			m_on_cuda = true;

			child_tensorcore->to_cuda("");

			//内部パラメータもCUDAに送る。
			weight->to_cuda("");
			bias->to_cuda("");
		}
		m_init_finish = true;
	}

	{
		if (input_tensorcore.mBatchSize != m_batch_size)
		{
			std::cout << "input batch size does not match" << std::endl;
			exit(1);
		}
		if (input_tensorcore.mCHW != m_input_size)
		{
			std::cout << "input chw size does not match" << std::endl;
			exit(1);
		}
	}

	{
		const auto& child_tensorcore = *m_child_tensorcore_tbl[0];
		const auto& weight = *m_parameter_tbl[0];
		const auto& bias = *m_parameter_tbl[1];


		//std::cout << "Affine forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
		if (m_on_cuda)
		{
			auto output_address = child_tensorcore._m_gpu_data_address;
			auto input_address = input_tensorcore._m_gpu_data_address;
			auto weight_address = weight._m_gpu_data_address;
			auto bias_address = bias._m_gpu_data_address;

			dim3 block(32, 32);
			dim3 grid((m_output_size + block.x - 1) / block.x, (m_batch_size + block.y - 1) / block.y);
			affine_forward_gpu_impl << <grid, block >> > (
				output_address,
				input_address,
				weight_address,
				bias_address,
				m_batch_size,
				m_output_size,
				m_input_size);
			CUDA_SYNCHRONIZE_DEBUG;
		}
		else
		{
			affine_forward_cpu_impl(input_tensors);
		}
	}


	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
}


void AffineCore::backward()
{
	//std::cout << "Affine backward" << std::endl;
	if (std::shared_ptr<TensorCore> input_tensorcore = mInputTensorCoreTbl[0].lock())
	{
		auto dataSize = m_child_tensorcore_tbl[0]->mDataSize;
		auto output_grad_address = m_child_tensorcore_tbl[0]->_m_gpu_grad_data_address;
		auto input_address = input_tensorcore->_m_gpu_data_address;
		auto input_grad_address = input_tensorcore->_m_gpu_grad_data_address;
		auto weight_grad_address = m_parameter_tbl[0]->_m_gpu_grad_data_address;
		auto bias_grad_address = m_parameter_tbl[1]->_m_gpu_grad_data_address;

		//パラメータの逆伝搬
		{
			if (m_on_cuda)
			{

				//Weight
				{
					dim3 block(16, 16);
					dim3 grid((m_input_size + block.x - 1) / block.x, (m_output_size + block.y - 1) / block.y);
					affine_backward_gpu_impl_weight << <grid, block >> > (
						weight_grad_address,
						output_grad_address,
						input_address,
						m_batch_size,
						m_output_size,
						m_input_size);
				}

				//Bias
				{
					dim3 block(16);
					dim3 grid((m_output_size + block.x - 1) / block.x);
					affine_backward_gpu_impl_bias << <grid, block >> >(
						bias_grad_address,
						output_grad_address,
						m_batch_size,
						m_output_size);
				}
				CUDA_SYNCHRONIZE_DEBUG;
			}
			else
			{
				affine_backward_cpu_impl_parameter(input_tensorcore);
			}
		}

		if (input_tensorcore->_m_need_grad)//勾配不要の場合、逆伝搬はスキップ出来る。
		{
			auto dataSize = m_child_tensorcore_tbl[0]->mDataSize;
			if (m_on_cuda)
			{
				dim3 block(16, 16);
				dim3 grid((m_input_size + block.x - 1) / block.x, (m_batch_size + block.y - 1) / block.y);
				affine_backward_gpu_impl_input << <grid, block >> > (
					input_grad_address,
					output_grad_address,
					weight_grad_address,
					m_batch_size,
					m_output_size,
					m_input_size);
				CUDA_SYNCHRONIZE_DEBUG;
			}
			else
			{
				affine_backward_cpu_impl_input(input_tensorcore);
			}
		}
	}
	else
	{
		std::cout << "Resource0 Error@ReLUCore::backward" << std::endl;
		exit(1);
	}
}



void AffineCore::affine_forward_cpu_impl(const LayerCore::iotype& input_tensors)
{
	const auto& input = *getTensorCoreFrom(input_tensors[0]);
	auto& output = *m_child_tensorcore_tbl[0];
	const auto& weight = *m_parameter_tbl[0];
	const auto& bias = *m_parameter_tbl[1];

	for (u32 N = 0, end = output.mBatchSize; N < end; N++)
	{
		for (u32 O = 0; O < m_output_size; O++)
		{
			u32 index = N * m_output_size + O;
			DataType result = 0.0f;
			for (u32 I = 0; I < m_input_size; I++)
			{
				result += weight(O, I) * input(N, I);
			}

			output(N, O) = result + bias(O);
		}
	}
}

void AffineCore::affine_backward_cpu_impl_input(const std::shared_ptr<TensorCore>& input_tensorcore)
{
	const auto& output = *m_child_tensorcore_tbl[0];
	auto& input = *input_tensorcore;
	auto& weight = *m_parameter_tbl[0];
	auto& bias = *m_parameter_tbl[1];

	for (u32 N = 0; N < m_batch_size; N++)
	{
		for (u32 I = 0; I < m_input_size; I++)
		{
			DataType result = 0.0f;
			for (u32 O = 0; O < m_output_size; O++)
			{
				result += weight(O, I) * output.d(N, O);
			}
			input.d(N, I) = result;
		}
	}
}

void AffineCore::affine_backward_cpu_impl_parameter(const std::shared_ptr<TensorCore>& input_tensorcore)
{
	const auto& output = *m_child_tensorcore_tbl[0];
	auto& input = *input_tensorcore;
	auto& weight = *m_parameter_tbl[0];
	auto& bias = *m_parameter_tbl[1];

	for (u32 O = 0; O < m_output_size; O++)
	{
		for (u32 I = 0; I < m_input_size; I++)
		{
			DataType result = 0.0f;
			for (u32 N = 0; N < m_batch_size; N++)
			{
				result += input(N, I) * output.d(N, O);
			}
			weight.d(O, I) = result;
		}

		DataType result = 0.0f;
		for (u32 N = 0; N < m_batch_size; N++)
		{
			result += output.d(N, O);
		}
		bias.d(O) = result;
	}
}