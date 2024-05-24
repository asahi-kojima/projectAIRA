#pragma once
#include "TensorCore.h"


/// <summary>
/// ���[�U�[�ɂƂ��ăe���\�����������߂̃C���^�[�t�F�[�X
/// </summary>
class Tensor
{
public:
	friend class LayerCore;
	Tensor() : pTensorCore(std::make_shared<TensorCore>())
	{}
	Tensor(const std::shared_ptr<TensorCore>& tensorCore) : pTensorCore(tensorCore)
	{}

	void backward()
	{
		pTensorCore->backward();
	}

	void setName(const std::string& name)//debug
	{
		pTensorCore->setName(name);
	}

	TensorCore::DataType getData(u32 index) const
	{
		return pTensorCore->getData(index);
	}

	u32 getDataSize() const
	{
		return pTensorCore->getDataSize();
	}

private:
	std::shared_ptr<TensorCore> pTensorCore;
};