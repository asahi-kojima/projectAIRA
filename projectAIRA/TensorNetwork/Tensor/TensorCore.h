#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include "typeinfo.h"
#include <cassert>

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

class LayerCore;
class Accessor2TensorCore;
class Tensor;

class TensorCore
{
public:
	friend class Tensor;
	friend class LayerCore;
	friend class Accessor2TensorCore;

	friend class AddCore;
	friend class SplitCore;
	friend class ReLUCore;

	//����͉�������������Ȃ��Ƃ����Ȃ��B
	TensorCore(){}

	TensorCore(const TensorCore& tensorcore, bool need_grad = false)
		: shape_already_set(tensorcore.shape_already_set)
		, mDimension(tensorcore.mDimension)
		, mBatchSize(tensorcore.mBatchSize)
		, mChannel(tensorcore.mChannel)
		, mHeight(tensorcore.mHeight)
		, mWidth(tensorcore.mWidth)
		, mDataSize(tensorcore.mDataSize)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim1)
		, mBatchSize(0)
		, mChannel(0)
		, mHeight(0)
		, mWidth(width)
		, mDataSize(width)
		, mCHW(0)
		, mHW(0)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//�f�[�^�T�C�Y����Ŋm�肵���̂ŁA����ɏ]���Ċm�ۂ���B
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim2)
		, mBatchSize(batchSize)
		, mChannel(0)
		, mHeight(0)
		, mWidth(width)
		, mDataSize(batchSize* width)
		, mCHW(0)
		, mHW(0)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//�f�[�^�T�C�Y����Ŋm�肵���̂ŁA����ɏ]���Ċm�ۂ���B
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 height, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim3)
		, mBatchSize(batchSize)
		, mChannel(0)
		, mHeight(height)
		, mWidth(width)
		, mDataSize(batchSize* height* width)
		, mCHW(0)
		, mHW(height* width)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//�f�[�^�T�C�Y����Ŋm�肵���̂ŁA����ɏ]���Ċm�ۂ���B
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}

	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool need_grad = false)

		: shape_already_set(true)
		, mDimension(Dimension::dim4)
		, mBatchSize(batchSize)
		, mChannel(channel)
		, mHeight(height)
		, mWidth(width)
		, mDataSize(batchSize* channel* height* width)
		, mCHW(channel * height * width)
		, mHW(height* width)

		, _m_need_grad(need_grad)
		, _m_on_cuda(false)
	{
		//�f�[�^�T�C�Y����Ŋm�肵���̂ŁA����ɏ]���Ċm�ۂ���B
		mallocOnCPU(_m_cpu_data_address, mDataSize);
		if (_m_need_grad)
		{
			mallocOnCPU(_m_cpu_grad_data_address, mDataSize);
		}
	}



	virtual ~TensorCore();

	//�e���\���̃f�[�^��GPU�ɓ]������B
	//GPU��̃������m�ہ��]��
	void to_cuda(const std::string&);

	//�t�`���֐����Ă�
	void callBackward() const;




	void regist_parent_layercore(const std::shared_ptr<LayerCore>&);

	void set_parent_exist(bool);

	void setName(const std::string&);


	DataType operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType operator()(u32, u32) const;
	DataType& operator()(u32, u32);



private:
	/*0000000000000000000000000000000000000000000000000000000000000000000*/
	//�`��Ɋւ���ϐ�
	enum class Dimension
	{
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
	bool _m_init_gpu_resource = false;/*GPU���\�[�X���m�ۂ�����*/

	DataType* _m_cpu_data_address = nullptr;/*�f�[�^��CPU�A�h���X*/
	DataType* _m_gpu_data_address = nullptr;/*�f�[�^��GPU�A�h���X*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/


	//NN�w�ƃe���\���̘A���Ɋւ���ϐ�
	bool m_parent_exist = false;/*�w�ɕR�Â��ꍇ�̂�true�ɂȂ�A�����̏ꍇ�̂݁A�t�`��������B*/

	//�e��c�����Ă���
	//backward�̏����ŕK�v�B
	std::weak_ptr<LayerCore> _m_upstream_layer;
	//s32 _m_location_in_upstream_layer = -1;

	//�����w�̏��
	//����������w�ɃC���v�b�g���ꂽ���ɁA�ǂ̑w�̉��Ԗڂ̃C���v�b�g��
	// �������ꂽ����o�^���Ă����B
	std::shared_ptr<LayerCore> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	/*1111111111111111111111111111111111111111111111111111111111111111111*/

	void synchronize_from_GPU_to_CPU();

	void disconnect_bidirection();
	void connect(const std::shared_ptr<LayerCore>&, u32);
	//���ʗp�̖��O�F�f�o�b�O�Ŏg�����Ƃ�z��
	std::string _m_debug_name;





	void deleteArrayAddress(DataType* p);
	void cuda_free(DataType* p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);






};
