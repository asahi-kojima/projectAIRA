#pragma once
#include "TensorCore.h"


class Tensor
{
public:
	friend class LayerCore;
	Tensor() : pTensorCore(std::make_shared<TensorCore>())
	{}

	void backward()
	{
		pTensorCore->backward();
	}

	void setName(const std::string& name)//debug
	{
		pTensorCore->setName(name);
	}
private:
	std::shared_ptr<TensorCore> pTensorCore;
	void fun() {}
};