#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include "typeinfo.h"
#include <cassert>

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

class LayerCore;
class Accessor2TensorCore;
class Tensor;

class TensorCore
{
public:
	friend class Tensor;
	friend class LayerCore;
	friend class Accessor2TensorCore;

	friend class AddCore;
	friend class SplitCore;
	friend class ReLUCore;

	//これは何かしら実装しないといけない。
	TensorCore(){}

	TensorCore(const TensorCore& tensorcore, bool need_grad = false)
		: shape_already_set(tensorcore.shape_already_set)
		, mDimension(tensorcore.mDimension)
		, mBatchSize(tensorcore.mBatchSize)
		, mChannel(tensorcore.mChannel)
		, mHeight(tensorcore.mHeight)
		, mWidth(tensorcore.mWidth)
		, mDataSize(tensorcore.mDataSize)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim1)
		, mBatchSize(0)
		, mChannel(0)
		, mHeight(0)
		, mWidth(width)
		, mDataSize(width)
		, mCHW(0)
		, mHW(0)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim2)
		, mBatchSize(batchSize)
		, mChannel(0)
		, mHeight(0)
		, mWidth(width)
		, mDataSize(batchSize* width)
		, mCHW(0)
		, mHW(0)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 height, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim3)
		, mBatchSize(batchSize)
		, mChannel(0)
		, mHeight(height)
		, mWidth(width)
		, mDataSize(batchSize* height* width)
		, mCHW(0)
		, mHW(height* width)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim4)
		, mBatchSize(batchSize)
		, mChannel(channel)
		, mHeight(height)
		, mWidth(width)
		, mDataSize(batchSize* channel* height* width)
		, mCHW(channel * height * width)
		, mHW(height* width)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}



	virtual ~TensorCore();

	//テンソルのデータをGPUに転送する。
	//GPU上のメモリ確保＆転送
	void to_cuda(const std::string&);

	//逆伝搬関数を呼ぶ
	void callBackward() const;




	void regist_parent_layercore(const std::shared_ptr<LayerCore>&);

	void set_parent_exist(bool);

	void setName(const std::string&);


	DataType operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType operator()(u32, u32) const;
	DataType& operator()(u32, u32);



private:
	/*0000000000000000000000000000000000000000000000000000000000000000000*/
	//形状に関する変数
	enum class Dimension
	{
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
	bool _m_init_gpu_resource = false;/*GPUリソースを確保したか*/

	DataType* _m_cpu_data_address = nullptr;/*データのCPUアドレス*/
	DataType* _m_gpu_data_address = nullptr;/*データのGPUアドレス*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/


	//NN層とテンソルの連結に関する変数
	bool m_parent_exist = false;/*層に紐づく場合のみtrueになり、かつその場合のみ、逆伝搬が走る。*/

	//親を把握しておく
	//backwardの処理で必要。
	std::weak_ptr<LayerCore> _m_upstream_layer;
	//s32 _m_location_in_upstream_layer = -1;

	//下流層の情報
	//自分がある層にインプットされた時に、どの層の何番目のインプットに
	// 結合されたかを登録しておく。
	std::shared_ptr<LayerCore> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	/*1111111111111111111111111111111111111111111111111111111111111111111*/

	void synchronize_from_GPU_to_CPU();

	void disconnect_bidirection();
	void connect(const std::shared_ptr<LayerCore>&, u32);
	//識別用の名前：デバッグで使うことを想定
	std::string _m_debug_name;





	void deleteArrayAddress(DataType* p);
	void cuda_free(DataType* p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);






};
