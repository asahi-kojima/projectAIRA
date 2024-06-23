#include <random>
#include "Layer.h"
#include "Convolution.h"


namespace
{
	__global__ void forward_gpu_impl_reshape(
		DataType* reshapedData,
		const DataType* input,
		aoba::nn::layer::ConvolutionCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::ConvolutionCore::parameterInfo& info = *pInfo;

		const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mIcIhIw = info.IcIhIw;

		if (N >= mBatchSize || IcIhIw >= mIcIhIw)
		{
			return;
		}


		//parameterInfoで決まるもの
		const u32 mFh = info.Fh;
		const u32 mFw = info.Fw;
		const u32 mFhFw = info.FhFw;
		const u32 mSh = info.Sh;
		const u32 mSw = info.Sw;

		const u32 mIw = info.Iw;
		const u32 mIcFhFw = info.IcFhFw;
		const u32 mIhIw = info.IhIw;

		const u32 mOh = info.Oh;
		const u32 mOw = info.Ow;

		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;


		//計算で決まるもの
		const u32 Ic = IcIhIw / mIhIw;
		const u32 Ih = (IcIhIw - Ic * mIhIw) / mIw;
		const u32 Iw = IcIhIw % mIw;

		const u32 exH = Ih + info.Ph;
		const u32 exW = Iw + info.Pw;





		DataType value = input[N * mIcIhIw + IcIhIw];

		for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / mSh), endOh = min(1 + (exH / mSh), mOh); Oh < endOh; Oh++)
		{
			for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / mSw), endOw = min(1 + (exW / mSw), mOw); Ow < endOw; Ow++)
			{
				const u32 row = Oh * mOw + Ow;
				const u32 col = Ic * mFhFw + (exH - Oh * mSh) * mFw + (exW - Ow * mSw);
				reshapedData[N * mOhOwIcFhFw + row * mIcFhFw + col] = value;
			}
		}
	}


	__global__ void forward_gpu_impl(
		DataType* y,
		const DataType* reshapedInput,
		const DataType* weight,
		const DataType* bias,
		aoba::nn::layer::ConvolutionCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::ConvolutionCore::parameterInfo& info = *pInfo;

		const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;

		if (N >= mBatchSize || OcOhOw >= mOcOhOw)
		{
			return;
		}

		const u32 id = N * mOcOhOw + OcOhOw;

		const u32 mOhOw = info.OhOw;
		const u32 mIcFhFw = info.IcFhFw;
		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;

		const u32 mFc = OcOhOw / mOhOw;
		const u32 OhOw = OcOhOw - mFc * mOhOw;

		f32 result = 0.0f;
		for (u32 i = 0; i < mIcFhFw; i++)
		{

			result += weight[mFc * mIcFhFw + i] * reshapedInput[N * mOhOwIcFhFw + OhOw * mIcFhFw + i];
		}

		y[id] = result + bias[mFc];
	}


	__global__ void backward_gpu_impl_input(
		DataType* input_grad,
		const DataType* output_grad,
		const DataType* weight,
		const aoba::nn::layer::ConvolutionCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::ConvolutionCore::parameterInfo& info = *pInfo;


		u32 N = blockIdx.x * blockDim.x + threadIdx.x;//input
		u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;//batch

		const u32 mBatchSize = info.batchSize;
		const u32 mIcIhIw = info.IcIhIw;
		if (N >= mBatchSize || IcIhIw >= mIcIhIw)
		{
			return;
		}

		u32 id = N * mIcIhIw + IcIhIw;

		const u32 mIhIw = info.IhIw;
		const u32 mIw = info.Iw;

		const u32 c = IcIhIw / mIhIw;
		const u32 h = (IcIhIw - c * mIhIw) / mIw;
		const u32 w = IcIhIw % mIw;

		const u32 exH = h + info.Ph;
		const u32 exW = w + info.Pw;

		const u32 mOh = info.Oh;
		const u32 mOw = info.Ow;
		const u32 mFh = info.Fh;
		const u32 mFw = info.Fw;
		const u32 mFhFw = info.FhFw;
		const u32 mFn = info.Fn;
		const u32 mSh = info.Sh;
		const u32 mSw = info.Sw;

		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOw = info.OhOw;
		const u32 mIcFhFw = info.IcFhFw;

		f32 result = 0.0f;
		for (u32 oh = (exH < mFh ? 0 : 1 + (exH - mFh) / mSh), endOh = min(1 + (exH / mSh), mOh); oh < endOh; oh++)
		{
			for (u32 ow = (exW < mFw ? 0 : 1 + (exW - mFw) / mSw), endOw = min(1 + (exW / mSw), mOw); ow < endOw; ow++)
			{
				const u32 row = oh * mOw + ow;
				const u32 col = c * mFhFw + (exH - oh * mSh) * mFw + (exW - ow * mSw);
				for (u32 Fc = 0; Fc < mFn; Fc++)
				{
					result += output_grad[N * mOcOhOw + Fc * mOhOw + row] * weight[Fc * mIcFhFw + col];
				}
			}
		}

		input_grad[id] = result;
	}

	__global__ void backward_gpu_impl_weight(
		DataType* weight_grad,
		const DataType* output_grad,
		const DataType* reshapedInput,
		const aoba::nn::layer::ConvolutionCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::ConvolutionCore::parameterInfo& info = *pInfo;

		u32 c = blockIdx.x * blockDim.x + threadIdx.x;
		u32 icfhfw = blockIdx.y * blockDim.y + threadIdx.y;
		if (c >= info.Oc || icfhfw >= info.IcFhFw)
		{
			return;
		}

		const u32 mBatchSize = info.batchSize;
		const u32 mOhOw = info.OhOw;
		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;
		const u32 mIcFhFw = info.IcFhFw;

		f32 result = 0.0f;
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 hw = 0; hw < mOhOw; hw++)
			{
				result += output_grad[N * mOcOhOw + c * mOhOw + hw] * reshapedInput[N * mOhOwIcFhFw + hw * mIcFhFw + icfhfw];
			}
		}
		weight_grad[c * mIcFhFw + icfhfw] = result;
	}

	__global__ void backward_gpu_impl_bias(
		DataType* bias_grad,
		const DataType* output_grad,
		const aoba::nn::layer::ConvolutionCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::ConvolutionCore::parameterInfo& info = *pInfo;

		u32 id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= info.Oc)
		{
			return;
		}

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOw = info.OhOw;

		f32 result = 0.0f;
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 hw = 0; hw < mOhOw; hw++)
			{
				result += output_grad[N * mOcOhOw + id * mOhOw + hw];
			}
		}

		bias_grad[id] = result;
	}
}



namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer Convolution(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight)
			{
				Layer conv = gen<ConvolutionCore>("Convolution", filterNum, filterSize, stride, padding, convWeight);
				return conv;
			}

			Layer Convolution(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight)
			{
				Layer conv = gen<ConvolutionCore>("Convolution", filterNum, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, convWeight);
				return conv;
			}



			ConvolutionCore::ConvolutionCore(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight)
				:ConvolutionCore(filterNum, filterSize, filterSize, stride, stride, padding, padding, convWeight)
			{

			}

			ConvolutionCore::ConvolutionCore(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight)
				:BaseLayer(1, 1, 1, 2)
				//入力非依存
				, mFn(filterNum)
				, mFh(filterHeight)
				, mFw(filterWidth)
				, mFhFw(filterHeight* filterWidth)
				, mSh(strideHeight)
				, mSw(strideWidth)
				, mPh(paddingHeight)
				, mPw(paddingWidth)
				, mConvolutionWeight(convWeight)
				//入力依存
				, mOutput(*m_output_tensorcore_tbl[0])
				, mWeight(*mTrainableParameterTbl[0])
				, mBias(*mTrainableParameterTbl[1])
				//ヘルパー
				, mReshapedInputData(false)
			{
				CHECK(cudaMalloc(&mParameterInfoOnGPU, sizeof(parameterInfo)));
			}

			ConvolutionCore::~ConvolutionCore()
			{
				CHECK(cudaFree(mParameterInfoOnGPU));
			}


			BaseLayer::iotype ConvolutionCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}


				const auto& input = *getTensorCoreFrom(input_tensors[0]);

				const bool input_on_cuda = input.isOnCuda();


				//入力の形状チェック
				{
					//入力が4次元でないと機能しない。これは仕様。
					if (input.getDimension() != TensorCore::Dimension::dim4)
					{
						assert(0);
					}
				}

				{
					//入力を参考に、変数の再設定を行う。
					resetVariable(input);

					mOutput.reshapeAs(mBatchSize, mOc, mOh, mOw, input_on_cuda);

					bool isInitReshapedInput = mReshapedInputData.reshapeExactlyAs(mBatchSize, mOhOw, mIcFhFw, input_on_cuda);

					//パラメータのreshape
					bool isWeightInit = mWeight.reshapeAs(mOc, mIcFhFw, input_on_cuda);
					bool isBiasInit = mBias.reshapeAs(mOc, input_on_cuda);


					//[修正済み]
					////この初期化は怪しい。mReshapedInputDataは計算時に特定の箇所に値を保存する。形状が同じであれば毎回同じ場所に値が入るので、それを利用して計算前の初期化をスキップしている。
					//// 一方でmReshapedInputDataをreshapeした際に、データサイズが一致していればメモリの再確保は行わない。
					////しかしデータサイズが一致しているが形状が異なる場合、仕様上初期化は行わないが、もしかしたら値をセットする箇所が異なるかもしれない。
					////その為、もし挙動がおかしい場合は順伝搬処理の前にmReshapedInputDataの初期化を入れる工程を毎回入れた方がいい。
					if (isInitReshapedInput)
					{
#ifdef _DEBUG
						std::cout << "Convolution Reshaped Param was initialized." << std::endl;
#endif // _DEBUG
						for (u32 i = 0, end = mReshapedInputData.getDataSize(); i < end; i++)
						{
							mReshapedInputData[i] = 0.0f;
						}
						mReshapedInputData.synchronize_from_CPU_to_GPU();
					}

					if (isWeightInit)
					{
#ifdef _DEBUG
						std::cout << "Convolution Weight Param was initialized." << std::endl;
#endif // _DEBUG
						std::random_device seed_gen;
						std::default_random_engine engine(seed_gen());
						std::normal_distribution<> dist(0.0f, std::sqrt(2.0f / mIcFhFw));
						for (u32 i = 0, end = mWeight.getDataSize(); i < end; i++)
						{
							mWeight[i] =  mConvolutionWeight* static_cast<DataType>(dist(engine));
						}
						mWeight.synchronize_from_CPU_to_GPU();
					}

					if (isBiasInit)
					{
#ifdef _DEBUG
						std::cout << "Convolution Bias Param was initialized." << std::endl;
#endif // _DEBUG
						for (u32 i = 0, end = mBias.getDataSize(); i < end; i++)
						{
							mBias[i] = 0.0f;
						}
						mBias.synchronize_from_CPU_to_GPU();
					}
				}


				{
					if (m_on_cuda)
					{
						auto output_gpu_address = mOutput.getGpuDataAddress();
						auto input_gpu_address = input.getGpuDataAddress();
						auto reshapedInput_gpu_address = mReshapedInputData.getGpuDataAddress();
						auto weight_gpu_address = mWeight.getGpuDataAddress();
						auto bias_gpu_address = mBias.getGpuDataAddress();

						//入力の形状変形
						{
							dim3 block(16, 16);
							dim3 grid(
								(mBatchSize + block.x - 1) / block.x,
								(mIcIhIw + block.y - 1) / block.y);

							forward_gpu_impl_reshape << <grid, block >> > (
								reshapedInput_gpu_address,
								input_gpu_address,
								mParameterInfoOnGPU);
						}

						//実際の順伝搬処理
						{
							dim3 block(16, 32);
							dim3 grid(
								(mBatchSize + block.x - 1) / block.x,
								(mOcOhOw + block.y - 1) / block.y);

							forward_gpu_impl << <grid, block >> > (
								output_gpu_address,
								reshapedInput_gpu_address,
								weight_gpu_address,
								bias_gpu_address,
								mParameterInfoOnGPU);
						}
					}
					else
					{
						forward_cpu_impl(input);
					}
				}

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}

			void ConvolutionCore::backward()
			{
				if (const std::shared_ptr<TensorCore>& input_ptr = mInputTensorCoreTbl[0].lock())
				{
					auto& input = *input_ptr;

					auto input_gpu_grad_address = input.getGpuGradDataAddress();
					const auto reshapedInput_gpu_address = mReshapedInputData.getGpuDataAddress();

					const auto output_gpu_grad_address = mOutput.getGpuGradDataAddress();

					auto weight_gpu_address = mWeight.getGpuDataAddress();
					auto weight_gpu_grad_address = mWeight.getGpuGradDataAddress();

					auto bias_gpu_grad_address = mBias.getGpuGradDataAddress();


					//パラメータの逆伝搬
					{
						if (m_on_cuda)
						{
							//weight
							{
								dim3 block(16, 16);
								dim3 grid((mOc + block.x - 1) / block.x, (mIcFhFw + block.y - 1) / block.y);

								backward_gpu_impl_weight << <grid, block >> > (
									weight_gpu_grad_address,
									output_gpu_grad_address,
									reshapedInput_gpu_address,
									mParameterInfoOnGPU);
							}
							//bias
							{
								dim3 block(16);
								dim3 grid((mOc + block.x - 1) / block.x);
								backward_gpu_impl_bias << <grid, block >> > (
									bias_gpu_grad_address,
									output_gpu_grad_address,
									mParameterInfoOnGPU);
							}
						}
						else
						{
							backward_cpu_impl_parameter();
						}
					}

					if (input.requiresGrad())
					{
						if (m_on_cuda)
						{
							dim3 block(16, 16);
							dim3 grid((mBatchSize + block.x - 1) / block.x, (mIcIhIw + block.y - 1) / block.y);

							backward_gpu_impl_input << <grid, block >> > (
								input_gpu_grad_address,
								output_gpu_grad_address,
								weight_gpu_address,
								mParameterInfoOnGPU);
							CUDA_SYNCHRONIZE_DEBUG;
							input.synchronize_from_GPU_to_CPU();
						}
						else
						{
							backward_cpu_impl_input(input);
						}
					}
				}
			}



			void ConvolutionCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 Ic = 0; Ic < mIc; Ic++)
					{
						for (u32 Ih = 0; Ih < mIh; Ih++)
						{
							for (u32 Iw = 0; Iw < mIw; Iw++)
							{
								const u32 exH = Ih + mPh;
								const u32 exW = Iw + mPw;

								auto value = input(N, Ic, Ih, Iw);

								for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / mSh), endOh = std::min(1 + (exH / mSh), mOh); Oh < endOh; Oh++)
								{
									for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / mSw), endOw = std::min(1 + (exW / mSw), mOw); Ow < endOw; Ow++)
									{
										const u32 row = Oh * mOw + Ow;
										const u32 col = Ic * mFhFw + (exH - Oh * mSh) * mFw + (exW - Ow * mSw);
										mReshapedInputData(N, row, col) = value;
									}
								}
							}
						}
					}

					for (u32 OcOhOw = 0; OcOhOw < mOcOhOw; OcOhOw++)
					{
						f32 tmp = 0.0f;
						const u32 Fc = OcOhOw / mOhOw;
						const u32 OhOw = OcOhOw - Fc * mOhOw;

						for (u32 IcFhFw = 0; IcFhFw < mIcFhFw; IcFhFw++)
						{
							tmp += mWeight(Fc, IcFhFw) * mReshapedInputData(N, OhOw, IcFhFw);
						}
						mOutput(N, OcOhOw) = tmp + mBias[Fc];
					}
				}
			}

			void ConvolutionCore::backward_cpu_impl_input(TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
					{
						const u32 c = IcIhIw / mIhIw;
						const u32 h = (IcIhIw - c * mIhIw) / mIw;
						const u32 w = IcIhIw % mIw;

						const u32 exH = h + mPh;
						const u32 exW = w + mPw;

						f32 result = 0.0f;
						for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / mSh), endOh = std::min(1 + (exH / mSh), mOh); Oh < endOh; Oh++)
						{
							for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / mSw), endOw = std::min(1 + (exW / mSw), mOw); Ow < endOw; Ow++)
							{
								const u32 row = Oh * mOw + Ow;
								const u32 col = c * mFhFw + (exH - Oh * mSh) * mFw + (exW - Ow * mSw);
								for (u32 Fc = 0; Fc < mOc; Fc++)
								{
									result += mOutput.d(N, Fc, row) * mWeight(Fc, col);
								}
							}
						}
						input.d(N, IcIhIw) = result;
					}
				}
			}

			void ConvolutionCore::backward_cpu_impl_parameter()
			{
				for (u32 Oc = 0; Oc < mOc; Oc++)
				{
					//フィルター行列の逆伝搬
					{
						for (u32 IcFhFw = 0; IcFhFw < mIcFhFw; IcFhFw++)
						{
							f32 tmp = 0;

							for (u32 N = 0; N < mBatchSize; N++)
							{
								for (u32 hw = 0; hw < mOhOw; hw++)
								{
									tmp += mOutput.d(N, Oc, hw) * mReshapedInputData(N, hw, IcFhFw);
								}
							}
							mWeight.d(Oc, IcFhFw) = tmp;
						}
					}

					//バイアスの逆伝搬
					{
						f32 tmp = 0.0f;
						for (u32 N = 0; N < mBatchSize; N++)
						{
							for (u32 hw = 0; hw < mOhOw; hw++)
							{
								tmp += mOutput.d(N, Oc, hw);
							}
						}
						mBias.d(Oc) = tmp;
					}
				}
			}

		}
	}
}