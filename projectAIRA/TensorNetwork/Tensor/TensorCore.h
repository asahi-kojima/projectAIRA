#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include "typeinfo.h"
#include <cassert>
#include "Layer/Layer.h"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#ifdef _DEBUG
#define CUDA_SYNCHRONIZE_DEBUG CHECK(cudaDeviceSynchronize())
#else
#define CUDA_SYNCHRONIZE_DEBUG {}
#endif



namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class Layer;
		}

		namespace optimizer
		{
			class Optimizer;
		}

		namespace tensor
		{
			class TensorCore;
		}
	}
}



class aoba::nn::tensor::TensorCore
{
public:
	friend class tensor::Tensor;
	friend class layer::Layer;

	//primitiveなクラスは速度の為に特権的にフレンドにしている。

	friend class optimizer::Optimizer;

	//これは何かしら実装しないといけない。
	TensorCore() {}

	TensorCore(const TensorCore& tensorcore, bool need_grad = false);

	TensorCore(u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 height, u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false);

	//TensorCore(Dimension, u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false);

	virtual ~TensorCore();

	//テンソルのデータをGPUに転送する。
	//GPU上のメモリ確保＆転送
	void to_cuda(const std::string&);

	//逆伝搬関数を呼ぶ
	void callBackward() const;

	bool getOnCuda() const { return _m_on_cuda; }



	DataType  operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType  operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType  operator()(u32, u32) const;
	DataType& operator()(u32, u32);
	DataType  operator()(u32) const;
	DataType& operator()(u32);

	DataType  d(u32, u32, u32, u32) const;
	DataType& d(u32, u32, u32, u32);
	DataType  d(u32, u32, u32) const;
	DataType& d(u32, u32, u32);
	DataType  d(u32, u32) const;
	DataType& d(u32, u32);
	DataType  d(u32) const;
	DataType& d(u32);

private:
	/*0000000000000000000000000000000000000000000000000000000000000000000*/
		//形状に関する変数
	enum class Dimension
	{
		dim0,
		dim1,
		dim2,
		dim3,
		dim4,
	};

	bool shape_already_set = false;/*形状を設定したかどうか*/
	Dimension mDimension;/*何次元データかを表現*/
	u32 mBatchSize;/*バッチサイズ*/
	u32 mChannel;/*チャンネル数*/
	u32 mHeight;/*高さ*/
	u32 mWidth;/*横幅*/
	u32 mDataSize;/*テンソルのデータサイズ*/
	u32 mCHW;
	u32 mHW;


	//CPU/GPUリソースに関する変数
	bool _m_need_grad = false;/*勾配が必要か。末端などでは必要ない。*/
	bool _m_on_cuda = false;/*GPUを利用するか否か*/

	DataType* _m_cpu_data_address = nullptr;/*データのCPUアドレス*/
	DataType* _m_gpu_data_address = nullptr;/*データのGPUアドレス*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/


	//NN層とテンソルの連結に関する変数
	bool m_parent_exist = false;/*層に紐づく場合のみtrueになり、かつその場合のみ、逆伝搬が走る。*/

	//親を把握しておく
	//backwardの処理で必要。
	std::weak_ptr<layer::Layer::LayerSkeleton> _m_upstream_layer;
	//s32 _m_location_in_upstream_layer = -1;

	//下流層の情報
	//自分がある層にインプットされた時に、どの層の何番目のインプットに
	// 結合されたかを登録しておく。
	std::shared_ptr<layer::Layer::LayerSkeleton> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	bool backward_finish = false;
	/*1111111111111111111111111111111111111111111111111111111111111111111*/

	void synchronize_from_GPU_to_CPU();
	void synchronize_from_CPU_to_GPU();
	void disconnect_bidirection();
	void connect(const std::shared_ptr<layer::Layer::LayerSkeleton>&, u32);
	//識別用の名前：デバッグで使うことを想定
	std::string _m_debug_name;





	void deleteArrayAddress(DataType* p);
	void cuda_free(DataType* p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);
	void mallocAndInitOnCPU(DataType*& pointer_on_cpu, const u32 element_num, const DataType initValue = 0.0f);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);
	void mallocAndInitOnGPU(DataType*& pointer_on_gpu, const u32 element_num, const DataType initValue = 0.0f);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);



	void regist_parent_layercore(const std::shared_ptr<layer::Layer::LayerSkeleton>&);
public:
	static void memcpyFromVector(Tensor&, const std::vector<DataType>&);
};

