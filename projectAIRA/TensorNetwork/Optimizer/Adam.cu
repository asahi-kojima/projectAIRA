#include "Layer/Layer.h"
#include "Adam.h"

namespace
{
	__global__ void optimize_gpu_impl(
		DataType* param,
		DataType* momentum ,
		DataType* velocity,  
		const DataType* param_grad, 
		const u32 dataSize, 
		const DataType effectiveLR, 
		const DataType beta0,
		const DataType beta1)
	{
		const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dataSize)
		{
			return;
		}

		const DataType paramGradValue = param_grad[index];
		const DataType tmpM = (momentum[index] += (1 - beta0) * (paramGradValue - momentum[index]) );
		const DataType tmpV = (velocity[index] += (1 - beta1) * (paramGradValue * paramGradValue - velocity[index]) );

		param[index] -= effectiveLR * tmpM / (sqrt(tmpV) + 1e-7);
	}
}


namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			Adam::Adam(DataType learningRate, DataType beta0, DataType beta1)
				:BaseOptimizer(learningRate)
				, mBeta0(beta0)
				, mBeta1(beta1)
			{
			}

			Adam::~Adam()
			{
			}


			void Adam::initialize()
			{
#ifdef _DEBUG
				std::cout << "Adam::initialize()" << std::endl;
#endif

				for (auto iter = mOptimizeScheduledParamTbl.begin(), end = mOptimizeScheduledParamTbl.end(); iter != end; iter++)
				{
					if (const std::shared_ptr<tensor::TensorCore>& parameter_ptr = (*iter).lock())
					{
						//テンソルと番号をマップで関係付ける。
						{
							bool isKeyExist = ((mMomentumMap.count(parameter_ptr.get()) == 1) && (mVelocityMap.count(parameter_ptr.get()) == 1));
							if (isKeyExist)
							{
								std::cout << "Key already exist in map. Duplicat" << std::endl;
								exit(1);
							}
							//mMomentumMap[parameter_ptr.get()] = tensor::TensorCore(false);
							//mVelocityMap[parameter_ptr.get()] = order;
							auto parameter_row_ptr = parameter_ptr.get();
							mMomentumMap.insert(std::make_pair(parameter_row_ptr, tensor::TensorCore(false)));
							mVelocityMap.insert(std::make_pair(parameter_row_ptr, tensor::TensorCore(false)));
							mIterationTbl.insert(std::make_pair(parameter_row_ptr, 0));

						}
					}
					else
					{
						std::cout << "Resource Error@BaseOptimizer::optimize" << std::endl;
						exit(1);
					}
				}
			}

			void Adam::optimize_unique(tensor::TensorCore& parameter)
			{
				auto& momentum = mMomentumMap[&parameter];
				auto& velocity = mVelocityMap[&parameter];
				auto& iteration = mIterationTbl[&parameter];
				const u32 paramDataSize = parameter.getDataSize();

				bool isInitM = momentum.reshapeExactlyAs(parameter, parameter.isOnCuda());
				bool isInitV = velocity.reshapeExactlyAs(parameter, parameter.isOnCuda());
				if (isInitM != isInitV)
				{
					assert(0);
				}

				if (isInitM)
				{
					iteration = 0;
					for (u32 i = 0, end = paramDataSize; i < end; i++)
					{
						momentum[i] = 0.0f;
						velocity[i] = 0.0f;
					}
					momentum.synchronize_from_CPU_to_GPU();
					velocity.synchronize_from_CPU_to_GPU();
				}
				iteration++;

				const DataType effectiveLR = mLearningRate * std::sqrtf(1.0f - std::powf(mBeta1, iteration)) / (1.0f - std::powf(mBeta0, iteration));
				if (parameter.isOnCuda())
				{
					auto param_gpu_address = parameter.getGpuDataAddress();
					auto momentum_gpu_address = momentum.getGpuDataAddress();
					auto velocity_gpu_address = velocity.getGpuDataAddress();
					auto param_gpu_grad_address = parameter.getGpuGradDataAddress();
					dim3 block(32);
					dim3 grid((block.x + paramDataSize - 1) / block.x);
					optimize_gpu_impl << <grid, block >> > (
						param_gpu_address,
						momentum_gpu_address,
						velocity_gpu_address,
						param_gpu_grad_address,
						paramDataSize,
						effectiveLR,
						mBeta0,
						mBeta1);
					CUDA_SYNCHRONIZE_DEBUG;
				}
				else
				{
					for (u32 i = 0, end = paramDataSize; i < end; i++)
					{
						DataType& paramValue = parameter(i);
						const DataType& paramGradValue = parameter.d(i);

						DataType tmpM = momentum[i] += (1 - mBeta0) * (paramGradValue - momentum[i]);
						DataType tmpV = velocity[i] += (1 - mBeta1) * (paramGradValue * paramGradValue - velocity[i]);

						paramValue -= effectiveLR * tmpM / (std::sqrtf(tmpV) + 1e-7);
					}
				}
			}
		}
	}
}