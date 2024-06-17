#include "TensorCore.h"
#include "Tensor.h"
#include "Layer/Layer.h"

namespace
{
	bool index_in_range(const u32 index, const u32 lower, const u32 upper)
	{
		return ((index >= lower) && (index < upper));
	}
}


using TensorCore = aoba::nn::tensor::TensorCore;

TensorCore::TensorCore()
	: mDimension(Dimension::dim0)
	, mBatchSize(0)
	, mChannel(0)
	, mHeight(0)
	, mWidth(0)
	, mDataSize(0)
	, mCHW(0)
	, mHW(0)

	, _m_need_grad(false)
	, _m_on_cuda(false)

{
}

TensorCore::TensorCore(const TensorCore& tensorcore, bool need_grad)
	: mDimension(tensorcore.mDimension)
	, mBatchSize(tensorcore.mBatchSize)
	, mChannel(tensorcore.mChannel)
	, mHeight(tensorcore.mHeight)
	, mWidth(tensorcore.mWidth)
	, mDataSize(tensorcore.mDataSize)
	, mCHW(tensorcore.mChannel* tensorcore.mHeight* tensorcore.mWidth)
	, mHW(tensorcore.mHeight* tensorcore.mWidth)

	, _m_need_grad(need_grad)
	, _m_on_cuda(false)

{
	mallocOnCPU(_m_cpu_data_address, mDataSize);
	if (need_grad)
	{
		mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
	}
}

TensorCore::TensorCore(u32 width, bool need_grad)
	: mDimension(Dimension::dim1)
	, mBatchSize(1)
	, mChannel(1)
	, mHeight(1)
	, mWidth(width)
	, mDataSize(1 * 1 * 1 * width)
	, mCHW(1 * 1 * width)
	, mHW(1 * width)

	, _m_need_grad(need_grad)
	, _m_on_cuda(false)

{
	//データサイズが上で確定したので、それに従って確保する。
	mallocOnCPU(_m_cpu_data_address, mDataSize);
	if (_m_need_grad)
	{
		mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
	}
}

TensorCore::TensorCore(u32 batchSize, u32 width, bool need_grad)
	: mDimension(Dimension::dim2)
	, mBatchSize(batchSize)
	, mChannel(1)
	, mHeight(1)
	, mWidth(width)
	, mDataSize(batchSize * 1 * 1 * width)
	, mCHW(1 * 1 * width)
	, mHW(1 * width)

	, _m_need_grad(need_grad)
	, _m_on_cuda(false)

{
	//データサイズが上で確定したので、それに従って確保する。
	mallocOnCPU(_m_cpu_data_address, mDataSize);
	if (_m_need_grad)
	{
		mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
	}
}

TensorCore::TensorCore(u32 batchSize, u32 height, u32 width, bool need_grad)
	: mDimension(Dimension::dim3)
	, mBatchSize(batchSize)
	, mChannel(1)
	, mHeight(height)
	, mWidth(width)
	, mDataSize(batchSize * 1 * height * width)
	, mCHW(1 * height * width)
	, mHW(height* width)

	, _m_need_grad(need_grad)
	, _m_on_cuda(false)

{
	//データサイズが上で確定したので、それに従って確保する。
	mallocOnCPU(_m_cpu_data_address, mDataSize);
	if (_m_need_grad)
	{
		mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
	}
}

TensorCore::TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad)
	: mDimension(Dimension::dim4)
	, mBatchSize(batchSize)
	, mChannel(channel)
	, mHeight(height)
	, mWidth(width)
	, mDataSize(batchSize* channel* height* width)
	, mCHW(channel* height* width)
	, mHW(height* width)

	, _m_need_grad(need_grad)
	, _m_on_cuda(false)

{
	//データサイズが上で確定したので、それに従って確保する。
	mallocOnCPU(_m_cpu_data_address, mDataSize);
	if (_m_need_grad)
	{
		mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
	}
}


TensorCore::~TensorCore()
{
	cleanMemory();
}

TensorCore& TensorCore::operator=(TensorCore&& tensorcore)
{
	mDimension = tensorcore.mDimension;
	mBatchSize = tensorcore.mBatchSize;
	mChannel = tensorcore.mChannel;
	mHeight = tensorcore.mHeight;
	mWidth = tensorcore.mWidth;
	mDataSize = tensorcore.mDataSize;
	mCHW = tensorcore.mCHW;
	mHW = tensorcore.mHW;

	_m_need_grad = tensorcore._m_need_grad;
	_m_on_cuda = tensorcore._m_on_cuda;

	_m_cpu_data_address = tensorcore._m_cpu_data_address;
	tensorcore._m_cpu_data_address = nullptr;
	_m_gpu_data_address = tensorcore._m_gpu_data_address;
	tensorcore._m_gpu_data_address = nullptr;
	_m_cpu_grad_data_address = tensorcore._m_cpu_grad_data_address;
	tensorcore._m_cpu_grad_data_address = nullptr;
	_m_gpu_grad_data_address = tensorcore._m_gpu_grad_data_address;
	tensorcore._m_gpu_grad_data_address = nullptr;

	m_parent_exist = tensorcore.m_parent_exist;


	_m_upstream_layer = tensorcore._m_upstream_layer;
	tensorcore._m_upstream_layer.reset();
	_m_downstream_layer = tensorcore._m_downstream_layer;
	tensorcore._m_downstream_layer.reset();
	_m_location_in_upstream_layer = tensorcore._m_location_in_upstream_layer;
	_m_location_in_downstream_layer = tensorcore._m_location_in_downstream_layer;

	return *this;
}

void TensorCore::resetShapeAs(u32 width)
{
	//未初期化の場合
	if (mDimension == Dimension::dim0)
	{
		*this = TensorCore(width);
	}
	else
	{

	}
}

void TensorCore::resetShapeAs(u32 batchSize, u32 channel, u32 height, u32 width)
{
	//未初期化の場合
	if (mDimension == Dimension::dim0)
	{

	}
	else
	{

	}
}



void TensorCore::to_cuda(const std::string& device_name)
{
	//cudaを使うことを記録しておく。
	if (_m_on_cuda)
	{
		//既にCUDAに転送済み。
		return;
	}

	_m_on_cuda = true;


	//CUDA用のメモリを確保する。
	//そのメモリにデータをコピーする。
	mallocOnGPU(_m_gpu_data_address, mDataSize);
	memcpyFromCPUToGPU(_m_cpu_data_address, _m_gpu_data_address, mDataSize);
	if (_m_need_grad)
	{
		mallocOnGPU(_m_gpu_grad_data_address, mDataSize);
		memcpyFromCPUToGPU(_m_cpu_grad_data_address, _m_gpu_grad_data_address, mDataSize);
	}
}

void TensorCore::callBackward() const
{
	if (std::shared_ptr<layer::Layer::LayerSkeleton> parentLayerCore = _m_upstream_layer.lock())
	{
		parentLayerCore->callBackward(_m_location_in_upstream_layer);
	}
	else if (m_parent_exist)
	{
		std::cout << "Resource error@TensorCore" << std::endl;
		exit(1);
	}
	else
	{
		std::cout << "no parent" << std::endl;
	}
}



void TensorCore::regist_upstream_layer(const std::shared_ptr<layer::Layer::LayerSkeleton>& parent_layercore)
{
	_m_upstream_layer = parent_layercore;
	m_parent_exist = true;
}



DataType TensorCore::operator()(u32 batchSize, u32 channel, u32 height, u32 width) const
{
	DataType* address = _m_cpu_data_address;

	const u32 index = batchSize * mCHW + channel * mHW + height * mWidth + width;
#ifdef _DEBUG
	bool condition0 = (mDimension == Dimension::dim4);
	bool condition1 = (batchSize < mBatchSize && channel < mChannel && height < mHeight && width < mWidth);
	bool condition2 = index_in_range(index, 0, mDataSize);
	if (!(condition0 && condition1 && condition2))
	{
		assert(0);
	}
#endif
	return address[index];
}
DataType& TensorCore::operator()(u32 batchSize, u32 channel, u32 height, u32 width)
{
	DataType* address = _m_cpu_data_address;

	const u32 index = batchSize * mCHW + channel * mHW + height * mWidth + width;
#ifdef _DEBUG
	bool condition0 = (mDimension == Dimension::dim4);
	bool condition1 = (batchSize < mBatchSize && channel < mChannel && height < mHeight && width < mWidth);
	bool condition2 = index_in_range(index, 0, mDataSize);
	if (!(condition0 && condition1 && condition2))
	{
		assert(0);
	}
#endif
	return address[index];
}

DataType TensorCore::operator()(u32 batchSize, u32 height, u32 width) const
{
	DataType* address = _m_cpu_data_address;

	//dim4の実装は過去のDeepLearningCppの実装を参考にした。
	if (mDimension == Dimension::dim4)
	{
		const u32 c = height;
		const u32 hw = width;
		const u32 index = batchSize * mCHW + c * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && c < mChannel && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 index = batchSize * mHW + height * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && height < mHeight && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}
DataType& TensorCore::operator()(u32 batchSize, u32 height, u32 width)
{
	DataType* address = _m_cpu_data_address;

	//dim4の実装は過去のDeepLearningCppの実装を参考にした。
	if (mDimension == Dimension::dim4)
	{
		const u32 c = height;
		const u32 hw = width;
		const u32 index = batchSize * mCHW + c * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && c < mChannel && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 index = batchSize * mHW + height * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && height < mHeight && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}

DataType TensorCore::operator()(u32 batchSize, u32 width) const
{
	DataType* address = _m_cpu_data_address;

	if (mDimension == Dimension::dim4)
	{
		const u32 chw = width;
		const u32 index = batchSize * mCHW + chw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && chw < mCHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 hw = width;
		const u32 index = batchSize * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim2)
	{
		const u32 index = batchSize * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}
DataType& TensorCore::operator()(u32 batchSize, u32 width)
{
	DataType* address = _m_cpu_data_address;

	if (mDimension == Dimension::dim4)
	{
		const u32 chw = width;
		const u32 index = batchSize * mCHW + chw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && chw < mCHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 hw = width;
		const u32 index = batchSize * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim2)
	{
		const u32 index = batchSize * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}

DataType TensorCore::operator()(u32 index) const
{
	DataType* address = _m_cpu_data_address;

	if (mDimension == Dimension::dim0)
	{
		assert(0);
	}
	else
	{
		if (!index_in_range(index, 0, mDataSize))
		{
			assert(0);
		}
		return address[index];
	}
}
DataType& TensorCore::operator()(u32 index)
{
	DataType* address = _m_cpu_data_address;

	if (mDimension == Dimension::dim0)
	{
		assert(0);
	}
	else
	{
		if (!index_in_range(index, 0, mDataSize))
		{
			assert(0);
		}
		return address[index];
	}
}



DataType  TensorCore::d(u32 batchSize, u32 channel, u32 height, u32 width) const
{
	DataType* address = _m_cpu_grad_data_address;

	const u32 index = batchSize * mCHW + channel * mHW + height * mWidth + width;
#ifdef _DEBUG
	bool condition0 = (mDimension == Dimension::dim4);
	bool condition1 = (batchSize < mBatchSize && channel < mChannel && height < mHeight && width < mWidth);
	bool condition2 = index_in_range(index, 0, mDataSize);
	if (!(condition0 && condition1 && condition2))
	{
		assert(0);
	}
#endif
	return address[index];
}
DataType& TensorCore::d(u32 batchSize, u32 channel, u32 height, u32 width)
{
	DataType* address = _m_cpu_grad_data_address;

	const u32 index = batchSize * mCHW + channel * mHW + height * mWidth + width;
#ifdef _DEBUG
	bool condition0 = (mDimension == Dimension::dim4);
	bool condition1 = (batchSize < mBatchSize && channel < mChannel && height < mHeight && width < mWidth);
	bool condition2 = index_in_range(index, 0, mDataSize);
	if (!(condition0 && condition1 && condition2))
	{
		assert(0);
	}
#endif
	return address[index];
}

DataType  TensorCore::d(u32 batchSize, u32 height, u32 width) const
{
	DataType* address = _m_cpu_grad_data_address;

	//dim4の実装は過去のDeepLearningCppの実装を参考にした。
	if (mDimension == Dimension::dim4)
	{
		const u32 c = height;
		const u32 hw = width;
		const u32 index = batchSize * mCHW + c * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && c < mChannel && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 index = batchSize * mHW + height * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && height < mHeight && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}
DataType& TensorCore::d(u32 batchSize, u32 height, u32 width)
{
	DataType* address = _m_cpu_grad_data_address;

	//dim4の実装は過去のDeepLearningCppの実装を参考にした。
	if (mDimension == Dimension::dim4)
	{
		const u32 c = height;
		const u32 hw = width;
		const u32 index = batchSize * mCHW + c * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && c < mChannel && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 index = batchSize * mHW + height * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && height < mHeight && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}

DataType  TensorCore::d(u32 batchSize, u32 width) const
{
	DataType* address = _m_cpu_grad_data_address;

	if (mDimension == Dimension::dim4)
	{
		const u32 chw = width;
		const u32 index = batchSize * mCHW + chw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && chw < mCHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 hw = width;
		const u32 index = batchSize * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim2)
	{
		const u32 index = batchSize * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}
DataType& TensorCore::d(u32 batchSize, u32 width)
{
	DataType* address = _m_cpu_grad_data_address;

	if (mDimension == Dimension::dim4)
	{
		const u32 chw = width;
		const u32 index = batchSize * mCHW + chw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && chw < mCHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim3)
	{
		const u32 hw = width;
		const u32 index = batchSize * mHW + hw;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && hw < mHW);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else if (mDimension == Dimension::dim2)
	{
		const u32 index = batchSize * mWidth + width;
#ifdef _DEBUG
		bool condition0 = (batchSize < mBatchSize && width < mWidth);
		bool condition1 = index_in_range(index, 0, mDataSize);
		if (!(condition0 && condition1))
		{
			assert(0);
		}
#endif
		return address[index];
	}
	else
	{
		assert(0);
	}
}

DataType  TensorCore::d(u32 index) const
{
	DataType* address = _m_cpu_grad_data_address;

	if (mDimension == Dimension::dim0)
	{
		assert(0);
	}
	else
	{
		if (!index_in_range(index, 0, mDataSize))
		{
			assert(0);
		}
		return address[index];
	}
}
DataType& TensorCore::d(u32 index)
{
	DataType* address = _m_cpu_grad_data_address;

	if (mDimension == Dimension::dim0)
	{
		assert(0);
	}
	else
	{
		if (!index_in_range(index, 0, mDataSize))
		{
			assert(0);
		}
		return address[index];
	}
}

void TensorCore::disconnect_bidirection()
{
	if (_m_downstream_layer)
	{
		_m_downstream_layer->disconnect_upstream_tensorcore(_m_location_in_downstream_layer);
		_m_downstream_layer.reset();
		_m_location_in_downstream_layer = -1;
		m_parent_exist = false;
	}
}

void TensorCore::synchronize_from_GPU_to_CPU()
{
	if (!_m_on_cuda)
		return;
	CHECK(cudaMemcpy(_m_cpu_data_address, _m_gpu_data_address, mDataSize * sizeof(DataType), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(_m_cpu_grad_data_address, _m_gpu_grad_data_address, mDataSize * sizeof(DataType), cudaMemcpyDeviceToHost));
	CUDA_SYNCHRONIZE_DEBUG;
}

void TensorCore::synchronize_from_CPU_to_GPU()
{
	if (!_m_on_cuda)
		return;
	CHECK(cudaMemcpy(_m_gpu_data_address, _m_cpu_data_address, mDataSize * sizeof(DataType), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(_m_gpu_grad_data_address, _m_cpu_grad_data_address, mDataSize * sizeof(DataType), cudaMemcpyHostToDevice));
	CUDA_SYNCHRONIZE_DEBUG;
}

void TensorCore::connect(const std::shared_ptr<layer::Layer::LayerSkeleton>& layercore, u32 location)
{
	_m_downstream_layer = layercore;
	_m_location_in_downstream_layer = location;
}

void TensorCore::cleanMemory()
{
	deleteArrayAddress(_m_cpu_data_address);
	deleteArrayAddress(_m_cpu_grad_data_address);
	cuda_free(_m_gpu_data_address);
	cuda_free(_m_gpu_grad_data_address);
}

void TensorCore::deleteArrayAddress(DataType*& p)
{
	delete[] p;
	p = nullptr;
}

void TensorCore::cuda_free(DataType*& p)
{
	if (p)
	{
		cudaFree(p);
		CUDA_SYNCHRONIZE_DEBUG;
	}
	p = nullptr;
}

void TensorCore::mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num)
{
	pointer_on_cpu = new DataType[element_num];
}

void TensorCore::mallocAndInitOnCPU(DataType*& pointer_on_cpu, const u32 element_num, const DataType initValue)
{
	pointer_on_cpu = new DataType[element_num];
	for (u32 i = 0; i < element_num; i++)
	{
		pointer_on_cpu[i] = initValue;
	}
}

void TensorCore::mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num)
{
	CHECK(cudaMalloc((void**)(&pointer_on_gpu), element_num * sizeof(DataType)););
	CUDA_SYNCHRONIZE_DEBUG;
}

void TensorCore::mallocAndInitOnGPU(DataType*& pointer_on_gpu, const u32 element_num, const DataType initValue)
{
	CHECK(cudaMalloc((void**)(&pointer_on_gpu), element_num * sizeof(DataType)););
	CUDA_SYNCHRONIZE_DEBUG;
	DataType* tmpMem = new DataType[element_num];
	for (u32 i = 0; i < element_num; i++)
	{
		tmpMem[i] = initValue;
	}
	CHECK(cudaMemcpy(pointer_on_gpu, tmpMem, element_num * sizeof(DataType), cudaMemcpyHostToDevice));
	CUDA_SYNCHRONIZE_DEBUG;
	delete[] tmpMem;
}

void TensorCore::memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size)
{
	CHECK(cudaMemcpy(gpu_address, cpu_address, data_size * sizeof(DataType), cudaMemcpyHostToDevice));
	CUDA_SYNCHRONIZE_DEBUG;
}


void TensorCore::memcpyFromVector(Tensor& tensor, const std::vector<DataType>& vec)
{
	auto& tensorcore = tensor;
	if (tensorcore.getDataSize() != vec.size())
	{
		assert(0);
	}

	for (u32 i = 0, end = tensorcore.getDataSize(); i < end; i++)
	{
		tensorcore(i) = vec[i];
	}
}

