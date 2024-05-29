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

class LayerCore;
class Accessor2TensorCore;
class Tensor;

class TensorCore
{
public:
	friend class Tensor;
	friend class LayerCore;
	friend class Accessor2TensorCore;

	using DataType = f32;
	using DataAddress = DataType*;


	TensorCore(bool need_grad, const std::vector<u32> shape_tbl)
		: mTensorShape(0)
		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//00000000000000000000000000000000000000000000000000000000
		//引数を解析して、テンソルの形状とデータサイズを登録する。
		const u32 dim = shape_tbl.size();
		u32 dataSize = 1;
		for (u32 i = 0; i < dim; i++)
		{
			const auto& shape = shape_tbl[i];
			dataSize *= shape;
			mTensorShape.push_back(shape);
		}
		mDataSize = dataSize;
		//111111111111111111111111111111111111111111111111111111111


		//00000000000000000000000000000000000000000000000000000000
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		//111111111111111111111111111111111111111111111111111111111
	}

	template<typename ... Args>
	TensorCore(Args ... args)
		: mTensorShape(0)
		, _m_need_grad(false)
		, _m_on_cuda(false)
	{
		//00000000000000000000000000000000000000000000000000000000
		//引数を解析して、テンソルの形状とデータサイズを登録する。
		u32 shape_tbl[] = { args ... };
		const u32 dim = sizeof(shape_tbl) / sizeof(shape_tbl[0]);
		u32 dataSize = 1;
		for (u32 i = 0; i < dim; i++)
		{
			const auto& shape = shape_tbl[i];
			dataSize *= shape;
			mTensorShape.push_back(shape);
		}
		mDataSize = dataSize;
		//111111111111111111111111111111111111111111111111111111111


		//00000000000000000000000000000000000000000000000000000000
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		//111111111111111111111111111111111111111111111111111111111
	}

	template<typename ... Args>
	TensorCore(bool need_grad, Args ... args)
		: mTensorShape(0)
		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//00000000000000000000000000000000000000000000000000000000
		//引数を解析して、テンソルの形状とデータサイズを登録する。
		u32 shape_tbl[] = { args ... };
		const u32 dim = sizeof(shape_tbl) / sizeof(shape_tbl[0]);
		u32 dataSize = 1;
		for (u32 i = 0; i < dim; i++)
		{
			const auto& shape = shape_tbl[i];
			dataSize *= shape;
			mTensorShape.push_back(shape);
		}
		mDataSize = dataSize;
		//111111111111111111111111111111111111111111111111111111111


		//00000000000000000000000000000000000000000000000000000000
		//データサイズが上で確定したので、それに従って確保する。
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
		//111111111111111111111111111111111111111111111111111111111
	}
	

	TensorCore() : _m_need_grad(false) {}
	virtual ~TensorCore();

	//テンソルのデータをGPUに転送する。
	//GPU上のメモリ確保＆転送
	void to_cuda(const std::string& device_name);

	//逆伝搬関数を呼ぶ
	void callBackward() const;


	DataType& getData(u32 index) 
	{
		if (m_parent_exist)
		{
			assert(0);
		}
		return _m_cpu_data_address[index]; 
	}
	DataType getData(u32 index) const { return _m_cpu_data_address[index]; }

	void regist_parent_layercore(const std::shared_ptr<LayerCore>& parent_layercore)
	{
		_m_upstream_layer = parent_layercore;
	}

	void set_parent_exist(bool parent_exist) { m_parent_exist = parent_exist; }
	
	void setName(const std::string& name) { _m_debug_name = name; }

	std::vector<u32> getShape() const
	{
		return mTensorShape;
	}

private:
	void disconnect_bidirection();
	void connect(const std::shared_ptr<LayerCore>&, u32);
	//識別用の名前：デバッグで使うことを想定
	std::string _m_debug_name;


	//データが置いてあるアドレス
	DataType* _m_cpu_data_address = nullptr;
	DataType* _m_gpu_data_address = nullptr;

	//勾配情報。以下のフラグが立っている場合のみ確保する
	DataType* _m_cpu_grad_data_address = nullptr;
	DataType* _m_gpu_grad_data_address = nullptr;
	//勾配が必要か。末端などでは必要ない。
	bool _m_need_grad;

	//テンソルのデータサイズ
	u32 mDataSize;
	//テンソルの形状(リスト形式で保持)
	std::vector<u32> mTensorShape;

	//GPUを利用するか否か
	bool _m_on_cuda;

	//親を把握しておく
	//backwardの処理で必要。
	std::weak_ptr<LayerCore> _m_upstream_layer;
	s32 _m_location_in_upstream_layer = -1;

	std::shared_ptr<LayerCore> _m_downstream_layer;
	s32 _m_location_in_downstram_layer = -1;
	//層に紐づく場合のみtrueになり、
	//かつその場合のみ、逆伝搬が走る。
	bool m_parent_exist = false;


	void deleteArrayAddress(DataType* p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);
};

using DataType = TensorCore::DataType;