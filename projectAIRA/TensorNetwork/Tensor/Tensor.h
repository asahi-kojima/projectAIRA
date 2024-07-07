#pragma once
#include "TensorCore.h"
//#include "Layer/BaseLayer.h"


/// <summary>
/// ユーザーにとってテンソルを扱うためのインターフェース
/// </summary>
namespace aoba { namespace nn { namespace tensor { class Tensor; } } }
class aoba::nn::tensor::Tensor
{
public:
	friend class layer::BaseLayer;

	Tensor();
	Tensor(const Tensor& tensor);
	Tensor(const std::shared_ptr<TensorCore>&);
	Tensor(s32 batchSize, s32 channel, s32 height, s32 width, bool need_grad = false);
	Tensor(s32 batchSize, s32 height, s32 width, bool need_grad = false);
	Tensor(s32 batchSize, s32 width, bool need_grad = false);
	~Tensor();

	Tensor& operator=(const Tensor&);
	Tensor& operator=(Tensor&&);

	void backward();

	u32 getDataSize() const;


	void to_cuda(bool);
	void synchronize_from_GPU_to_CPU();
	void synchronize_from_CPU_to_GPU();
	

	DataType operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType operator()(u32, u32) const;
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
	std::shared_ptr<TensorCore> pTensorCore;
};
