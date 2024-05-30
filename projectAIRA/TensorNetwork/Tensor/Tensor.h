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

	Tensor() : pTensorCore(std::make_shared<TensorCore>()) {}
	Tensor(const Tensor& tensor) : pTensorCore(tensor.pTensorCore){}
	Tensor(const std::shared_ptr<TensorCore>& tensorCore) : pTensorCore(tensorCore){}
	template<typename ... Args>
	Tensor(Args ... args)
		: pTensorCore(std::make_shared<TensorCore>(args...))
	{

	}

	void backward()
	{
		pTensorCore->callBackward();
	}

	void setName(const std::string& name)//debug
	{
		pTensorCore->setName(name);
	}

	u32 getTensorDataSize() const
	{
		return pTensorCore->mDataSize;
	}

	std::vector<u32> getShape() const
	{
		return pTensorCore->getShape();
	}


	DataType operator[](u32 index) const
	{
		return pTensorCore->_m_cpu_data_address[index];
	}
	
	DataType& operator[](u32 index)
	{
		return pTensorCore->_m_cpu_data_address[index];
	}


	void to_cuda()
	{
		pTensorCore->to_cuda("");
	}

private:
	std::shared_ptr<TensorCore> pTensorCore;
};
