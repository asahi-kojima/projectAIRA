#include "Tensor.h"

namespace aoba::nn::tensor
{
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

	Tensor::Tensor(s32 batchSize, s32 channel, s32 height, s32 width, bool need_grad)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize),
			static_cast<u32>(channel),
			static_cast<u32>(height),
			static_cast<u32>(width),
			need_grad))
	{
	}

	Tensor::Tensor(s32 batchSize, s32 height, s32 width, bool need_grad)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize),
			static_cast<u32>(height),
			static_cast<u32>(width),
			need_grad))
	{
	}

	Tensor::Tensor(s32 batchSize, s32 width, bool need_grad)
		:pTensorCore(std::make_shared<TensorCore>(
			static_cast<u32>(batchSize),
			static_cast<u32>(width),
			need_grad))
	{
	}

	void Tensor::backward()
	{
		pTensorCore->backward_finish = true;
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


	void Tensor::to_cuda(bool gpu_is_available)
	{
		if (gpu_is_available)
			pTensorCore->to_cuda("");
	}

	void Tensor::synchronize_from_GPU_to_CPU()
	{
		pTensorCore->synchronize_from_GPU_to_CPU();
	}


	DataType Tensor::operator()(u32 batchSize, u32 channel, u32 height, u32 width) const
	{
		return (*pTensorCore)(batchSize, channel, height, width);
	}
	DataType& Tensor::operator()(u32 batchSize, u32 channel, u32 height, u32 width)
	{
		return (*pTensorCore)(batchSize, channel, height, width);
	}
	DataType Tensor::operator()(u32 batchSize, u32 height, u32 width) const
	{
		return (*pTensorCore)(batchSize, height, width);
	}
	DataType& Tensor::operator()(u32 batchSize, u32 height, u32 width)
	{
		return (*pTensorCore)(batchSize, height, width);
	}
	DataType Tensor::operator()(u32 batchSize, u32 width) const
	{
		return (*pTensorCore)(batchSize, width);
	}
	DataType& Tensor::operator()(u32 batchSize, u32 width)
	{
		return (*pTensorCore)(batchSize, width);
	}
	DataType Tensor::operator()(u32 index) const
	{
		return (*pTensorCore)(index);
	}
	DataType& Tensor::operator()(u32 index)
	{
		return (*pTensorCore)(index);
	}

	DataType Tensor::d(u32 batchSize, u32 channel, u32 height, u32 width) const
	{
		return pTensorCore->d(batchSize, channel, height, width);
	}
	DataType& Tensor::d(u32 batchSize, u32 channel, u32 height, u32 width)
	{
		return pTensorCore->d(batchSize, channel, height, width);
	}
	DataType Tensor::d(u32 batchSize, u32 height, u32 width) const
	{
		return pTensorCore->d(batchSize, height, width);
	}
	DataType& Tensor::d(u32 batchSize, u32 height, u32 width)
	{
		return pTensorCore->d(batchSize, height, width);
	}
	DataType Tensor::d(u32 batchSize, u32 width) const
	{
		return pTensorCore->d(batchSize, width);
	}
	DataType& Tensor::d(u32 batchSize, u32 width)
	{
		return pTensorCore->d(batchSize, width);
	}
	DataType Tensor::d(u32 index) const
	{
		return pTensorCore->d(index);
	}
	DataType& Tensor::d(u32 index)
	{
		return pTensorCore->d(index);
	}

	DataType Tensor::getLoss(const Tensor& tensor)
	{
		if (tensor.pTensorCore->mDataSize != 1)
		{
			assert(0);
		}
		return tensor.pTensorCore->_m_cpu_data_address[0];
	}

}