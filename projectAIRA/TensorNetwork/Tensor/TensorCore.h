#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>
#include "typeinfo.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class BaseLayer;
			class Layer;
		}

		namespace optimizer
		{
			class Optimizer;
		}

		namespace tensor
		{
			class TensorCore;
			class Tensor;
		}
	}
}



class aoba::nn::tensor::TensorCore
{
public:
	friend class tensor::Tensor;
	friend class layer::BaseLayer;//NN層は出力テンソルを生成するが、特別な操作を要するため他に利用されないようにフレンド属性を与える。

	//形状に関する変数
	enum class Dimension
	{
		dim0,//未初期化（デフォルトコンストラクタ呼び出し）の場合のみこれに該当
		dim1,
		dim2,
		dim3,
		dim4,
	};

	/// <summary>
	/// 全てのメンバ変数を未初期化の状態に留める。
	/// </summary>
	TensorCore(bool grad_required = false);
	TensorCore(const TensorCore& tensorcore, bool grad_required = false, bool memory_synch = false);//要再検討：メモリをどこまでコピーするか
	TensorCore(u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 height, u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool grad_required = false);
	virtual ~TensorCore();

	
	bool reshapeAs(const TensorCore&, bool on_cuda = false);
	bool reshapeAs(u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 height, u32 width, bool on_cuda = false);
	bool reshapeExactlyAs(u32 batchSize, u32 height, u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 channel, u32 height, u32 width, bool on_cuda = false);


	//テンソルのデータをGPUに転送する。
	//GPU上のメモリ確保＆転送
	void to_cuda();

	//逆伝搬関数を呼ぶ
	void callBackward() const;




	DataType  operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType  operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType  operator()(u32, u32) const;
	DataType& operator()(u32, u32);
	DataType  operator()(u32) const;
	DataType& operator()(u32);
	DataType operator[](u32) const;
	DataType& operator[](u32);

	DataType  d(u32, u32, u32, u32) const;
	DataType& d(u32, u32, u32, u32);
	DataType  d(u32, u32, u32) const;
	DataType& d(u32, u32, u32);
	DataType  d(u32, u32) const;
	DataType& d(u32, u32);
	DataType  d(u32) const;
	DataType& d(u32);

	Dimension getDimension() const
	{
		return mDimension;
	}

	u32 getBatchSize() const
	{
		return mBatchSize;
	}

	u32 getChannel() const { return mChannel; }
	u32 getHeight() const { return mHeight; }
	u32 getWidth() const { return mWidth; }

	u32 getHW() const { return mHeight * mWidth; }

	u32 getCHW() const
	{
		return mCHW;
	}

	u32 getDataSize() const
	{
		return mDataSize;
	}

	bool isOnCuda() const
	{
		return m_on_cuda;
	}

	bool requiresGrad() const
	{
		return m_grad_required;
	}

	DataType* getGpuDataAddress()
	{
		return _m_gpu_data_address;
	}

	const DataType* getGpuDataAddress() const
	{
		return _m_gpu_data_address;
	}

	DataType* getGpuGradDataAddress()
	{
		return _m_gpu_grad_data_address;
	}

	const DataType* getGpuGradDataAddress() const
	{
		return _m_gpu_grad_data_address;
	}

	void synchronize_from_GPU_to_CPU();
	void synchronize_from_CPU_to_GPU();


private:
	/*0000000000000000000000000000000000000000000000000000000000000000000*/


	Dimension mDimension;/*何次元データかを表現*/
	u32 mBatchSize;/*バッチサイズ*/
	u32 mChannel;/*チャンネル数*/
	u32 mHeight;/*高さ*/
	u32 mWidth;/*横幅*/
	u32 mHW;/*高さ×横幅*/
	u32 mCHW;/*チャンネル×高さ×横幅*/
	u32 mDataSize;/*テンソルのデータサイズ*/


	//CPU/GPUリソースに関する変数
	bool m_grad_required = false;/*勾配が必要か。末端などでは必要ない。*/
	bool m_on_cuda = false;/*GPUを利用するか否か*/

	DataType* _m_cpu_data_address = nullptr;/*データのCPUアドレス*/
	DataType* _m_gpu_data_address = nullptr;/*データのGPUアドレス*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU勾配情報。on_cudaフラグが立っている場合のみ確保する*/


	//親(上流層)を把握しておく
	//backwardの処理で必要。
	std::weak_ptr<layer::BaseLayer> _m_upstream_layer;
	s32 _m_location_in_upstream_layer = -1;
	bool m_upstream_exist = false;/*層に紐づく場合のみtrueになり、かつその場合のみ、逆伝搬が走る。*/

	//下流層の情報
	//自分がある層にインプットされた時に、どの層の何番目のインプットに
	// 結合されたかを登録しておく。
	std::shared_ptr<layer::BaseLayer> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	/*1111111111111111111111111111111111111111111111111111111111111111111*/
	void disconnect_bidirection();
	void connect(const std::shared_ptr<layer::BaseLayer>&, u32);
	void regist_upstream_layer(const std::shared_ptr<layer::BaseLayer>&);

	bool isSameShape(const TensorCore&);
	bool isSameShape(Dimension, u32 batchSize, u32 channel, u32 height, u32 width);
	void setNewShape(const TensorCore&);
	void setNewShape(Dimension, u32 batchSize, u32 channel, u32 height, u32 width);

	TensorCore& operator=(TensorCore&& tensorcore);

	void cleanMemory();
	void deleteArrayAddress(DataType*& p);
	void cuda_free(DataType*& p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);
	void mallocAndInitOnCPU(DataType*& pointer_on_cpu, const u32 element_num, const DataType initValue = 0.0f);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);
	void mallocAndInitOnGPU(DataType*& pointer_on_gpu, const u32 element_num, const DataType initValue = 0.0f);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);



	//const DataType* get_cpu_data_address() const;
	//DataType* get_cpu_data_address();

public:
	static void memcpyFromVector(Tensor&, const std::vector<DataType>&);
};

