#include "Tensor.h"


Tensor::Tensor()
	: pTensorCore(std::make_shared<TensorCore>()) 
{
}

Tensor::Tensor(const Tensor& tensor)
	: pTensorCore(tensor.pTensorCore) 
{
}

Tensor::Tensor(const std::shared_ptr<TensorCore>& tensorCore)
	: pTensorCore(tensorCore) 
{
}


void Tensor::backward()
{
	pTensorCore->callBackward();
}

u32 Tensor::getDataSize() const
{
	return pTensorCore->mDataSize;
}

void Tensor::setName(const std::string& name)//debug
{
	pTensorCore->setName(name);
}


DataType Tensor::operator()(u32 batchSize, u32 channel, u32 height, u32 width) const
{
	return (*this)(batchSize, channel, height, width);
}
DataType& Tensor::operator()(u32 batchSize, u32 channel, u32 height, u32 width)
{
	return (*this)(batchSize, channel, height, width);
}
DataType Tensor::operator()(u32 batchSize, u32 height, u32 width) const
{
	return (*this)(batchSize, height, width);
}
DataType& Tensor::operator()(u32 batchSize, u32 height, u32 width)
{
	return (*this)(batchSize, height, width);
}
DataType Tensor::operator()(u32 batchSize, u32 width) const
{
	return (*this)(batchSize, width);
}
DataType& Tensor::operator()(u32 batchSize, u32 width)
{
	return (*this)(batchSize, width);
}
