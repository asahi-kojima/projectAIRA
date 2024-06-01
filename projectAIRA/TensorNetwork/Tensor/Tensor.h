#pragma once
#include "TensorCore.h"


/// <summary>
/// ユーザーにとってテンソルを扱うためのインターフェース
/// </summary>
class Tensor
{
public:
	friend class LayerCore;
	friend class Accessor2TensorCore;

	Tensor();
	Tensor(const Tensor& tensor);
	Tensor(const std::shared_ptr<TensorCore>&);

	Tensor(s32 batchSize, s32 channel, s32 height, s32 width, bool need_grad = false)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize), 
			static_cast<u32>(channel), 
			static_cast<u32>(height), 
			static_cast<u32>(width), 
			need_grad))
	{
	}

	Tensor(s32 batchSize, s32 height, s32 width, bool need_grad = false)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize),
			static_cast<u32>(height),
			static_cast<u32>(width),
			need_grad))
	{
	}

	Tensor(s32 batchSize,s32 width, bool need_grad = false)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize),
			static_cast<u32>(width),
			need_grad))
	{
	}


	void backward();

	u32 getDataSize() const;

	void setName(const std::string& name);//debug


	//DataType operator[](u32 index) const
	//{
	//	return pTensorCore->_m_cpu_data_address[index];
	//}

	//DataType& operator[](u32 index)
	//{
	//	return pTensorCore->_m_cpu_data_address[index];
	//}

	DataType operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType operator()(u32, u32) const;
	DataType& operator()(u32, u32);



	void to_cuda()
	{
		pTensorCore->to_cuda("");
	}
	void synchronize_from_GPU_to_CPU()
	{
		pTensorCore->synchronize_from_GPU_to_CPU();
	}
private:
	std::shared_ptr<TensorCore> pTensorCore;
};
