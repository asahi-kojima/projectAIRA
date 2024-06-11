#pragma once
#include "TensorCore.h"


/// <summary>
/// ユーザーにとってテンソルを扱うためのインターフェース
/// </summary>
class Accessor2TensorCore;
namespace aoba { namespace nn { namespace tensor { class Tensor; } } }
class aoba::nn::tensor::Tensor
{
public:
	friend class aoba::nn::layer::LayerCore;
	friend class TensorCore;
	friend class Accessor2TensorCore;

	Tensor();
	Tensor(const Tensor& tensor);
	Tensor(const std::shared_ptr<TensorCore>&);
	Tensor(s32 batchSize, s32 channel, s32 height, s32 width, bool need_grad = false);
	Tensor(s32 batchSize, s32 height, s32 width, bool need_grad = false);
	Tensor(s32 batchSize, s32 width, bool need_grad = false);


	void backward();

	u32 getDataSize() const;

	void setName(const std::string& name);//debug

	void to_cuda(bool);

	void synchronize_from_GPU_to_CPU();

	DataType operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType operator()(u32, u32) const;
	DataType& operator()(u32, u32);


	static DataType getLoss(const Tensor&);

private:
	std::shared_ptr<TensorCore> pTensorCore;
};
