#include <random>
#include "Layer.h"
#include "TransposeConv.h"


namespace
{
	__global__ void forward_gpu_impl_reshape(
		DataType* reshapedData,
		const DataType* input,
		aoba::nn::layer::TransposeConvCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::TransposeConvCore::parameterInfo& info = *pInfo;

		const u32 IcIhIw = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 N = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mIcIhIw = info.IcIhIw;

		if (N >= mBatchSize || IcIhIw >= mIcIhIw)
		{
			return;
		}


		//parameterInfoで決まるもの
		const u32 mFh = info.Fh;
		const u32 mFw = info.Fw;
		const u32 mFhFw = mFh * mFw;
		const u32 mSh = info.Sh;
		const u32 mSw = info.Sw;


		const u32 mIw = info.Iw;
		const u32 mIhIw = info.IhIw;
		const u32 mIcFhFw = info.IcFhFw;

		const u32 mOh = info.Oh;
		const u32 mOw = info.Ow;

		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;

		//計算で決まるもの
		const u32 Ic = IcIhIw / mIhIw;
		const u32 Ih = (IcIhIw - Ic * mIhIw) / mIw;
		const u32 Iw = IcIhIw % mIw;




		const u32 exH = mFh - 1 - info.Ph + Ih * mSh;
		const u32 exW = mFw - 1 - info.Pw + Iw * mSw;



		f32 value = input[N * mIcIhIw + IcIhIw];

		for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = min(1 + (exH / 1), mOh); Oh < endOh; Oh++)
		{
			for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = min(1 + (exW / 1), mOw); Ow < endOw; Ow++)
			{
				const u32 row = Oh * mOw + Ow;
				const u32 col = Ic * mFhFw + (exH - Oh * 1) * mFw + (exW - Ow * 1);
				reshapedData[N * mOhOwIcFhFw + row * mIcFhFw + col] = value;
			}
		}
	}


	__global__ void forward_gpu_impl(
		DataType* y,
		const DataType* reshapedInput,
		const DataType* weight,
		const DataType* bias,
		aoba::nn::layer::TransposeConvCore::parameterInfo* pInfo)
	{
		aoba::nn::layer::TransposeConvCore::parameterInfo& info = *pInfo;

		const u32 OcOhOw = blockIdx.x * blockDim.x + threadIdx.x;
		const u32 N = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;

		if (N >= mBatchSize || OcOhOw >= mOcOhOw)
		{
			return;
		}


		const u32 mOhOw = info.OhOw;
		const u32 mFc = OcOhOw / mOhOw;
		const u32 OhOw = OcOhOw - mFc * mOhOw;

		const u32 mIcFhFw = info.IcFhFw;
		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;

		f32 result = 0.0f;
		for (u32 i = 0; i < mIcFhFw; i++)
		{

			result += weight[mFc * mIcFhFw + i] * reshapedInput[N * mOhOwIcFhFw + OhOw * mIcFhFw + i];
		}

		y[N * mOcOhOw + OcOhOw] = result + bias[mFc];
	}


	__global__ void backward_gpu_impl_input(
		DataType* input_grad,
		const DataType* output_grad,
		const DataType* weight,
		const aoba::nn::layer::TransposeConvCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::TransposeConvCore::parameterInfo& info = *pInfo;


		const u32 IcIhIw = blockIdx.x * blockDim.x + threadIdx.x;//input
		const u32 N = blockIdx.y * blockDim.y + threadIdx.y;//batch

		const u32 mBatchSize = info.batchSize;
		const u32 mIcIhIw = info.IcIhIw;


		if (N >= mBatchSize || IcIhIw >= mIcIhIw)
		{
			return;
		}


		const u32 mFh = info.Fh;
		const u32 mFw = info.Fw;
		const u32 mFhFw = info.FhFw;
		const u32 mFn = info.Fn;
		const u32 mSh = info.Sh;
		const u32 mSw = info.Sw;

		const u32 mIw = info.Iw;
		const u32 mIhIw = info.IhIw;

		const u32 Ic = IcIhIw / mIhIw;
		const u32 Ih = (IcIhIw - Ic * mIhIw) / mIw;
		const u32 Iw = IcIhIw % mIw;

		const u32 mOh = info.Oh;
		const u32 mOw = info.Ow;

		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOw = info.OhOw;
		const u32 mIcFhFw = info.IcFhFw;

		const u32 exH = mFh - 1 - info.Ph + Ih * mSh;
		const u32 exW = mFw - 1 - info.Pw + Iw * mSw;

		f32 result = 0.0f;
		for (u32 oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = min(1 + (exH / 1), mOh); oh < endOh; oh++)
		{
			for (u32 ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = min(1 + (exW / 1), mOw); ow < endOw; ow++)
			{
				const u32 row = oh * mOw + ow;
				const u32 col = Ic * mFhFw + (exH - oh * 1) * mFw + (exW - ow * 1);
				for (u32 Fc = 0; Fc < mFn; Fc++)
				{
					result += output_grad[N * mOcOhOw + Fc * mOhOw + row] * weight[Fc * mIcFhFw + col];
				}
			}
		}


		input_grad[N * mIcIhIw + IcIhIw] = result;
	}

	__global__ void backward_gpu_impl_weight(
		DataType* weight_grad,
		const DataType* output_grad,
		const DataType* reshapedInput,
		const aoba::nn::layer::TransposeConvCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::TransposeConvCore::parameterInfo& info = *pInfo;

		u32 IcFhFw = blockIdx.x * blockDim.x + threadIdx.x;
		u32 Oc = blockIdx.y * blockDim.y + threadIdx.y;

		const u32 mIcFhFw = info.IcFhFw;

		if (Oc >= info.Oc || IcFhFw >= mIcFhFw)
		{
			return;
		}

		const u32 mBatchSize = info.batchSize;
		const u32 mOhOw = info.OhOw;
		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOwIcFhFw = info.OhOwIcFhFw;

		DataType result = 0.0f;
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 hw = 0; hw < mOhOw; hw++)
			{
				result += output_grad[N * mOcOhOw + Oc * mOhOw + hw] * reshapedInput[N * mOhOwIcFhFw + hw * mIcFhFw + IcFhFw];
			}
		}
		weight_grad[Oc * mIcFhFw + IcFhFw] = result;
	}

	__global__ void backward_gpu_impl_bias(
		DataType* bias_grad,
		const DataType* output_grad,
		const aoba::nn::layer::TransposeConvCore::parameterInfo* pInfo)
	{
		const aoba::nn::layer::TransposeConvCore::parameterInfo& info = *pInfo;

		u32 Oc = blockIdx.x * blockDim.x + threadIdx.x;
		if (Oc >= info.Oc)
		{
			return;
		}

		const u32 mBatchSize = info.batchSize;
		const u32 mOcOhOw = info.OcOhOw;
		const u32 mOhOw = info.OhOw;

		DataType result = 0.0f;
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 hw = 0; hw < mOhOw; hw++)
			{
				result += output_grad[N * mOcOhOw + Oc * mOhOw + hw];
			}
		}

		bias_grad[Oc] = result;
	}
}



namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			Layer TransposeConv(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 convWeight)
			{
				Layer transposeConv = gen<TransposeConvCore>("TransposeConv", filterNum, filterSize, stride, padding, convWeight);
				return transposeConv;
			}

			Layer TransposeConv(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 convWeight)
			{
				Layer transposeConv = gen<TransposeConvCore>("TransposeConv", filterNum, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, convWeight);
				return transposeConv;
			}



			TransposeConvCore::TransposeConvCore(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 weight)
				:TransposeConvCore(filterNum, filterSize, filterSize, stride, stride, padding, padding, weight)
			{
			}

			TransposeConvCore::TransposeConvCore(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 weight)
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
				, mTransposeConvWeight(weight)
				//入力依存
				, mOutput(*m_output_tensorcore_tbl[0])
				, mWeight(*mTrainableParameterTbl[0])
				, mBias(*mTrainableParameterTbl[1])
				//ヘルパー
				, mReshapedInputData(false)
			{
			}

			TransposeConvCore::~TransposeConvCore()
			{
				if (mIsParamerInfoAllocated)
				{
					CHECK(cudaFree(mParameterInfoOnGPU));
				}
			}


			BaseLayer::iotype TransposeConvCore::forward(const BaseLayer::iotype& input_tensors)
			{
				if (!m_init_finish)
				{
					initialize();
				}


				const auto& input = *getTensorCoreFrom(input_tensors[0]);


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

					mOutput.reshapeAs(mBatchSize, mOc, mOh, mOw, m_on_cuda);

					bool isInitReshapedInput = mReshapedInputData.reshapeExactlyAs(mBatchSize, mOhOw, mIcFhFw, m_on_cuda);

					//パラメータのreshape
					bool isWeightInit = mWeight.reshapeAs(mOc, mIcFhFw, m_on_cuda);
					bool isBiasInit = mBias.reshapeAs(mOc, m_on_cuda);


					//[修正済み]
					////この初期化は怪しい。mReshapedInputDataは計算時に特定の箇所に値を保存する。形状が同じであれば毎回同じ場所に値が入るので、それを利用して計算前の初期化をスキップしている。
					//// 一方でmReshapedInputDataをreshapeした際に、データサイズが一致していればメモリの再確保は行わない。
					////しかしデータサイズが一致しているが形状が異なる場合、仕様上初期化は行わないが、もしかしたら値をセットする箇所が異なるかもしれない。
					////その為、もし挙動がおかしい場合は順伝搬処理の前にmReshapedInputDataの初期化を入れる工程を毎回入れた方がいい。
					if (isInitReshapedInput)
					{
#ifdef _DEBUG
						std::cout << "TransposeConv Reshaped was initialized." << std::endl;
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
						std::cout << "TransposeConv Weight Param was initialized." << std::endl;
#endif // _DEBUG
						std::random_device seed_gen;
						std::default_random_engine engine(seed_gen());
						std::normal_distribution<> dist(0.0f, std::sqrt(2.0f / mIcFhFw));
						for (u32 i = 0, end = mWeight.getDataSize(); i < end; i++)
						{
							mWeight[i] = mTransposeConvWeight * static_cast<DataType>(dist(engine));
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
							dim3 block(32, 32);
							dim3 grid(
								(mIcIhIw + block.x - 1) / block.x,
								(mBatchSize + block.y - 1) / block.y);

							forward_gpu_impl_reshape << <grid, block >> > (
								reshapedInput_gpu_address,
								input_gpu_address,
								mParameterInfoOnGPU);
						}

						//実際の順伝搬処理
						{
							dim3 block(32, 32);
							dim3 grid(
								(mOcOhOw + block.x - 1) / block.x,
								(mBatchSize + block.y - 1) / block.y);
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							forward_gpu_impl << <grid, block >> > (
								output_gpu_address,
								reshapedInput_gpu_address,
								weight_gpu_address,
								bias_gpu_address,
								mParameterInfoOnGPU);
							CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_gpu_impl");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
					}
					else
					{
#ifdef TIME_DEBUG
						std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
						forward_cpu_impl(input);
#ifdef TIME_DEBUG
						f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
						std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "forward_cpu_impl");
						debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
					}
				}

				return iotype{ Tensor(m_output_tensorcore_tbl[0]) };
			}

			void TransposeConvCore::backward()
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
								dim3 block(32, 32);
								dim3 grid((mIcFhFw + block.x - 1) / block.x, (mOc + block.y - 1) / block.y);
#ifdef TIME_DEBUG
								std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
								backward_gpu_impl_weight << <grid, block >> > (
									weight_gpu_grad_address,
									output_gpu_grad_address,
									reshapedInput_gpu_address,
									mParameterInfoOnGPU);
								CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
								f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
								std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_gpu_impl_weight");
								debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
							}
							//bias
							{
								dim3 block(32);
								dim3 grid((mOc + block.x - 1) / block.x);
#ifdef TIME_DEBUG
								std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
								backward_gpu_impl_bias << <grid, block >> > (
									bias_gpu_grad_address,
									output_gpu_grad_address,
									mParameterInfoOnGPU);
								CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
								f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
								std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_gpu_impl_bias");
								debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
							}
						}
						else
						{
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_cpu_impl_parameter();
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_cpu_impl_parameter");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
					}

					if (input.requiresGrad())
					{
						if (m_on_cuda)
						{
#ifdef _DEBUG
							dim3 block(32, 16);
#else
							dim3 block(32, 32);
#endif
							dim3 grid((mIcIhIw + block.x - 1) / block.x, (mBatchSize + block.y - 1) / block.y);
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_gpu_impl_input << <grid, block >> > (
								input_gpu_grad_address,
								output_gpu_grad_address,
								weight_gpu_address,
								mParameterInfoOnGPU);
							CUDA_SYNCHRONIZE_DEBUG;
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_gpu_impl_input");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
						else
						{
#ifdef TIME_DEBUG
							std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif // TIME_DEBUG
							backward_cpu_impl_input(input);
#ifdef TIME_DEBUG
							f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
							std::string name = makeDebugIdentifier(mInstanceID, __FUNCTION__, "backward_cpu_impl_input");
							debugTimers[name] = elapsedTime;
#endif // TIME_DEBUG
						}
					}
				}
			}



			void TransposeConvCore::forward_cpu_impl(const TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 Ic = 0; Ic < mIc; Ic++)
					{
						for (u32 Ih = 0; Ih < mIh; Ih++)
						{
							for (u32 Iw = 0; Iw < mIw; Iw++)
							{
								const u32 exH = mFh - 1 - mPh + Ih * mSh;
								const u32 exW = mFw - 1 - mPw + Iw * mSw;

								const auto& value = input(N, Ic, Ih, Iw);

								for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = std::min(1 + (exH / 1), mOh); Oh < endOh; Oh++)
								{
									for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = std::min(1 + (exW / 1), mOw); Ow < endOw; Ow++)
									{
										const u32 row = Oh * mOw + Ow;
										const u32 col = Ic * mFhFw + (exH - Oh * 1) * mFw + (exW - Ow * 1);
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

			void TransposeConvCore::backward_cpu_impl_input(TensorCore& input)
			{
				for (u32 N = 0; N < mBatchSize; N++)
				{
					for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
					{
						const u32 c = IcIhIw / mIhIw;
						const u32 h = (IcIhIw - c * mIhIw) / mIw;
						const u32 w = IcIhIw % mIw;

						const u32 exH = mFh - 1 - mPh + h * mSh;
						const u32 exW = mFw - 1 - mPw + w * mSw;

						f32 result = 0.0f;
						for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = std::min(1 + (exH / 1), mOh); Oh < endOh; Oh++)
						{
							for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = std::min(1 + (exW / 1), mOw); Ow < endOw; Ow++)
							{
								const u32 row = Oh * mOw + Ow;
								const u32 col = c * mFhFw + (exH - Oh * 1) * mFw + (exW - Ow * 1);
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

			void TransposeConvCore::backward_cpu_impl_parameter()
			{
				for (u32 c = 0; c < mOc; c++)
				{
					//フィルター行列の逆伝搬
					{
						for (u32 icfhfw = 0; icfhfw < mIcFhFw; icfhfw++)
						{
							f32 tmp = 0;

							for (u32 N = 0; N < mBatchSize; N++)
							{
								for (u32 hw = 0; hw < mOhOw; hw++)
								{
									tmp += mOutput.d(N, c, hw) * mReshapedInputData(N, hw, icfhfw);
								}
							}
							mWeight.d(c, icfhfw) = tmp;
						}
					}

					//バイアスの逆伝搬
					{
						f32 tmp = 0.0f;
						for (u32 N = 0; N < mBatchSize; N++)
						{
							for (u32 hw = 0; hw < mOhOw; hw++)
							{
								tmp += mOutput.d(N, c, hw);
							}
						}
						mBias.d(c) = tmp;
					}
				}
			}

		}
	}
}