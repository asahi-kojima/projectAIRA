#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>
#include "typeinfo.h"


namespace aoba
{
	namespace nn
	{
		namespace layer
		{
			class BaseLayer;
			class Layer;
		}

		namespace optimizer
		{
			class Optimizer;
		}

		namespace tensor
		{
			class TensorCore;
			class Tensor;
		}
	}
}



class aoba::nn::tensor::TensorCore
{
public:
	friend class tensor::Tensor;
	friend class layer::BaseLayer;//NN�w�͏o�̓e���\���𐶐����邪�A���ʂȑ����v���邽�ߑ��ɗ��p����Ȃ��悤�Ƀt�����h������^����B

	//�`��Ɋւ���ϐ�
	enum class Dimension
	{
		dim0,//���������i�f�t�H���g�R���X�g���N�^�Ăяo���j�̏ꍇ�݂̂���ɊY��
		dim1,
		dim2,
		dim3,
		dim4,
	};

	/// <summary>
	/// �S�Ẵ����o�ϐ��𖢏������̏�Ԃɗ��߂�B
	/// </summary>
	TensorCore(bool grad_required = false);
	TensorCore(const TensorCore& tensorcore, bool grad_required = false, bool memory_synch = false);//�v�Č����F���������ǂ��܂ŃR�s�[���邩
	TensorCore(u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 height, u32 width, bool grad_required = false);
	TensorCore(u32 batchSize, u32 channel, u32 height, u32 width, bool grad_required = false);
	virtual ~TensorCore();

	
	bool reshapeAs(const TensorCore&, bool on_cuda = false);
	bool reshapeAs(u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 height, u32 width, bool on_cuda = false);
	bool reshapeExactlyAs(u32 batchSize, u32 height, u32 width, bool on_cuda = false);
	bool reshapeAs(u32 batchSize, u32 channel, u32 height, u32 width, bool on_cuda = false);


	//�e���\���̃f�[�^��GPU�ɓ]������B
	//GPU��̃������m�ہ��]��
	void to_cuda();

	//�t�`���֐����Ă�
	void callBackward() const;




	DataType  operator()(u32, u32, u32, u32) const;
	DataType& operator()(u32, u32, u32, u32);
	DataType  operator()(u32, u32, u32) const;
	DataType& operator()(u32, u32, u32);
	DataType  operator()(u32, u32) const;
	DataType& operator()(u32, u32);
	DataType  operator()(u32) const;
	DataType& operator()(u32);
	DataType operator[](u32) const;
	DataType& operator[](u32);

	DataType  d(u32, u32, u32, u32) const;
	DataType& d(u32, u32, u32, u32);
	DataType  d(u32, u32, u32) const;
	DataType& d(u32, u32, u32);
	DataType  d(u32, u32) const;
	DataType& d(u32, u32);
	DataType  d(u32) const;
	DataType& d(u32);

	Dimension getDimension() const
	{
		return mDimension;
	}

	u32 getBatchSize() const
	{
		return mBatchSize;
	}

	u32 getChannel() const { return mChannel; }
	u32 getHeight() const { return mHeight; }
	u32 getWidth() const { return mWidth; }

	u32 getHW() const { return mHeight * mWidth; }

	u32 getCHW() const
	{
		return mCHW;
	}

	u32 getDataSize() const
	{
		return mDataSize;
	}

	bool isOnCuda() const
	{
		return m_on_cuda;
	}

	bool requiresGrad() const
	{
		return m_grad_required;
	}

	DataType* getGpuDataAddress()
	{
		return _m_gpu_data_address;
	}

	const DataType* getGpuDataAddress() const
	{
		return _m_gpu_data_address;
	}

	DataType* getGpuGradDataAddress()
	{
		return _m_gpu_grad_data_address;
	}

	const DataType* getGpuGradDataAddress() const
	{
		return _m_gpu_grad_data_address;
	}

	void synchronize_from_GPU_to_CPU();
	void synchronize_from_CPU_to_GPU();


private:
	/*0000000000000000000000000000000000000000000000000000000000000000000*/


	Dimension mDimension;/*�������f�[�^����\��*/
	u32 mBatchSize;/*�o�b�`�T�C�Y*/
	u32 mChannel;/*�`�����l����*/
	u32 mHeight;/*����*/
	u32 mWidth;/*����*/
	u32 mHW;/*�����~����*/
	u32 mCHW;/*�`�����l���~�����~����*/
	u32 mDataSize;/*�e���\���̃f�[�^�T�C�Y*/


	//CPU/GPU���\�[�X�Ɋւ���ϐ�
	bool m_grad_required = false;/*���z���K�v���B���[�Ȃǂł͕K�v�Ȃ��B*/
	bool m_on_cuda = false;/*GPU�𗘗p���邩�ۂ�*/

	DataType* _m_cpu_data_address = nullptr;/*�f�[�^��CPU�A�h���X*/
	DataType* _m_gpu_data_address = nullptr;/*�f�[�^��GPU�A�h���X*/

	DataType* _m_cpu_grad_data_address = nullptr;/*CPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/
	DataType* _m_gpu_grad_data_address = nullptr;/*GPU���z���Bon_cuda�t���O�������Ă���ꍇ�̂݊m�ۂ���*/


	//�e(�㗬�w)��c�����Ă���
	//backward�̏����ŕK�v�B
	std::weak_ptr<layer::BaseLayer> _m_upstream_layer;
	s32 _m_location_in_upstream_layer = -1;
	bool m_upstream_exist = false;/*�w�ɕR�Â��ꍇ�̂�true�ɂȂ�A�����̏ꍇ�̂݁A�t�`��������B*/

	//�����w�̏��
	//����������w�ɃC���v�b�g���ꂽ���ɁA�ǂ̑w�̉��Ԗڂ̃C���v�b�g��
	// �������ꂽ����o�^���Ă����B
	std::shared_ptr<layer::BaseLayer> _m_downstream_layer;
	s32 _m_location_in_downstream_layer = -1;


	/*1111111111111111111111111111111111111111111111111111111111111111111*/
	void disconnect_bidirection();
	void connect(const std::shared_ptr<layer::BaseLayer>&, u32);
	void regist_upstream_layer(const std::shared_ptr<layer::BaseLayer>&);

	bool isSameShape(const TensorCore&);
	bool isSameShape(Dimension, u32 batchSize, u32 channel, u32 height, u32 width);
	void setNewShape(const TensorCore&);
	void setNewShape(Dimension, u32 batchSize, u32 channel, u32 height, u32 width);

	TensorCore& operator=(TensorCore&& tensorcore);

	void cleanMemory();
	void deleteArrayAddress(DataType*& p);
	void cuda_free(DataType*& p);

	void mallocOnCPU(DataType*& pointer_on_cpu, const u32 element_num);
	void mallocAndInitOnCPU(DataType*& pointer_on_cpu, const u32 element_num, const DataType initValue = 0.0f);

	void mallocOnGPU(DataType*& pointer_on_gpu, const u32 element_num);
	void mallocAndInitOnGPU(DataType*& pointer_on_gpu, const u32 element_num, const DataType initValue = 0.0f);

	void memcpyFromCPUToGPU(DataType* cpu_address, DataType* gpu_address, u32 data_size);



	//const DataType* get_cpu_data_address() const;
	//DataType* get_cpu_data_address();

public:
	static void memcpyFromVector(Tensor&, const std::vector<DataType>&);
};

