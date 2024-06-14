#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include "typeinfo.h"
#include <cassert>
#include "Layer/Layer.h"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#ifdef _DEBUG
#define CUDA_SYNCHRONIZE_DEBUG CHECK(cudaDeviceSynchronize())
#else
#define CUDA_SYNCHRONIZE_DEBUG {}
#endif



namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class Layer;
		}

		namespace optimizer
		{
			class Optimizer;
		}

		namespace tensor
		{
			class TensorCore;
		}
	}
}



class aoba::nn::tensor::TensorCore
{
public:
	friend class tensor::Tensor;
	friend class layer::Layer;

	//primitive�ȃN���X�͑��x�ׂ̈ɓ����I�Ƀt�����h�ɂ��Ă���B

	friend class optimizer::Optimizer;

	//����͉�������������Ȃ��Ƃ����Ȃ��B
	TensorCore() {}

	TensorCore(const TensorCore& tensorcore, bool need_grad = false);

	TensorCore(u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 height, u32 width, bool need_grad = false);

	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false);

	//TensorCore(Dimension, u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false);

	virtual ~TensorCore();

	//�e���\���̃f�[�^��GPU�ɓ]������B
	//GPU��̃������m�ہ��]��
	void to_cuda(const std::string&);

	//�t�`���֐����Ă�
	void callBackward() const;

	bool getOnCuda() const { return _m_on_cuda; }



	DataType  operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType  operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType  operator()(u32, u32) const;
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
	/*0000000000000000000000000000000000000000000000000000000000000000000*/
		//�`��Ɋւ���ϐ�
	enum class Dimension
	{
		dim0,
		dim1,
		dim2,
		dim3,
		dim4,
	};

	bool shape_already_set = false;/*�`���ݒ肵�����ǂ���*/
	Dimension mDimension;/*�������f�[�^����\��*/
	u32 mBatchSize;/*�o�b�`�T�C�Y*/
	u32 mChannel;/*�`�����l����*/
	u32 mHeight;/*����*/
	u32 mWidth;/*����*/
	u32 mDataSize;/*�e���\���̃f�[�^�T�C�Y*/
	u32 mCHW;
	u32 mHW;


	//CPU/GPU���\�[�X�Ɋւ���ϐ�
	bool _m_need_grad = false;/*���z���K�v���B���[�Ȃǂł͕K�v�Ȃ��B*/
	bool _m_on_cuda = false;/*GPU�𗘗p���邩�ۂ�*/

	DataType* _m_cpu_data_address = nullptr;/*�f�[�^��CPU�A�h���X*/
	DataType* _m_gpu_data_address = nullptr;/*�f�[�^��GPU�A�h���X*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/


	//NN�w�ƃe���\���̘A���Ɋւ���ϐ�
	bool m_parent_exist = false;/*�w�ɕR�Â��ꍇ�̂�true�ɂȂ�A�����̏ꍇ�̂݁A�t�`��������B*/

	//�e��c�����Ă���
	//backward�̏����ŕK�v�B
	std::weak_ptr<layer::Layer::LayerSkeleton> _m_upstream_layer;
	//s32 _m_location_in_upstream_layer = -1;

	//�����w�̏��
	//����������w�ɃC���v�b�g���ꂽ���ɁA�ǂ̑w�̉��Ԗڂ̃C���v�b�g��
	// �������ꂽ����o�^���Ă����B
	std::shared_ptr<layer::Layer::LayerSkeleton> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	bool backward_finish = false;
	/*1111111111111111111111111111111111111111111111111111111111111111111*/

	void synchronize_from_GPU_to_CPU();
	void synchronize_from_CPU_to_GPU();
	void disconnect_bidirection();
	void connect(const std::shared_ptr<layer::Layer::LayerSkeleton>&, u32);
	//���ʗp�̖��O�F�f�o�b�O�Ŏg�����Ƃ�z��
	std::string _m_debug_name;





	void deleteArrayAddress(DataType* p);
	void cuda_free(DataType* p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);
	void mallocAndInitOnCPU(DataType*& pointer_on_cpu, const u32 element_num, const DataType initValue = 0.0f);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);
	void mallocAndInitOnGPU(DataType*& pointer_on_gpu, const u32 element_num, const DataType initValue = 0.0f);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);



	void regist_parent_layercore(const std::shared_ptr<layer::Layer::LayerSkeleton>&);
public:
	static void memcpyFromVector(Tensor&, const std::vector<DataType>&);
};

