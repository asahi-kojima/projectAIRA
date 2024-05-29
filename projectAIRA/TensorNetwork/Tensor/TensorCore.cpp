#include "TensorCore.h"
#include "Layer/LayerCore.h"


TensorCore::~TensorCore()
{
	deleteArrayAddress(_m_cpu_data_address);
	deleteArrayAddress(_m_gpu_data_address);
	deleteArrayAddress(_m_cpu_grad_data_address);
	deleteArrayAddress(_m_gpu_grad_data_address);
}



void TensorCore::to_cuda(const std::string& device_name)
{
	//cudaを使うことを記録しておく。
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
	if (std::shared_ptr<LayerCore> parentLayerCore = _m_upstream_layer.lock())
	{
		parentLayerCore->callBackward();
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


void TensorCore::disconnect_bidirection()
{
	if (_m_downstream_layer)
	{
		_m_downstream_layer->disconnect_bidirection(_m_location_in_downstram_layer);
		_m_downstream_layer.reset();
		_m_location_in_downstram_layer = -1;
	}
}

void TensorCore::connect(const std::shared_ptr<LayerCore>& layercore, u32 location)
{
	_m_downstream_layer = layercore;
	_m_location_in_downstram_layer = location;
}


void TensorCore::deleteArrayAddress(DataType* p)
{
	if (p)
	{
		delete[] p;
	}
}

void TensorCore::mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num)
{
	pointer_on_cpu = new DataType[element_num];
}

void TensorCore::mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num)
{
	CHECK(cudaMalloc((void**)(pointer_on_gpu), element_num * sizeof(DataType)););
}

void TensorCore::memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size)
{
	CHECK(cudaMemcpy(gpu_address, cpu_address, data_size * sizeof(DataType), cudaMemcpyHostToDevice));
}