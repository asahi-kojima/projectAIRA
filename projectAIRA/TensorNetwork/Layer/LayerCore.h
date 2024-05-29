#pragma once
#include "Tensor/Tensor.h"


//�R���X�g���N�^�Ŏq�e���\����shared_ptr������this��o�^�������Čp���B
//��肪�N�����炱�����^���B
class LayerCore : public std::enable_shared_from_this<LayerCore>
{
public:
	friend class TensorCore;
	using iotype = std::vector<Tensor>;

	LayerCore(u32=1, u32=1);
	LayerCore(u32, u32, u32);
	virtual ~LayerCore() {}

	iotype callForward(const iotype&);
	void callBackward();
	void regist_this_to_output_tensor();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }


	/// <summary>
	/// �e�w���Ǝ��ɍs��forward�����͂��̉��z�֐��Ɏ�������B
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual iotype forward(const iotype& input_tensors) = 0;

	virtual iotype operator()(const iotype& input)
	{
		return forward(input);
	}

protected:
	bool m_init_finish = false;
	bool m_use_gpu = false;
	/// <summary>
	/// �e�w���Ǝ��ɍs��forward�����͂��̉��z�֐��Ɏ�������B
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual void backward()
	{
		std::cout << "no implement" << std::endl;
	}

	std::vector<std::shared_ptr<TensorCore> > m_parameter_tbl;

	/// <summary>
	/// ���̑w�����������e���\��
	///�i�e�w�̓e���\���p�̃������𒼐ڌ��Ă���C���[�W�j
	/// </summary>
	std::vector<std::shared_ptr<TensorCore>> m_child_tensorcore_tbl;

	/// <summary>
	/// ���`���ŃC���v�b�g���ꂽ�e���\�������o���Ă����p
	/// ���ꂪ�Ȃ��Ƌt�`���������ōs���Ȃ��Ȃ�B
	/// </summary>
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;


	//���͂Əo�͂̃e���\���̐����L�^
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;

private:
	void disconnect_bidirection(s32 location)
	{
		mInputTensorCoreTbl[location].reset();
	}
};

class Accessor2TensorCore
{
public:
	inline static TensorCore::DataType* getAddressOnCpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_cpu_data_address;
	}
	inline static TensorCore::DataType* getAddressOnCpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_cpu_data_address;
	}

	inline static TensorCore::DataType* getAddressOnGpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_gpu_data_address;
	}
	inline static TensorCore::DataType* getAddressOnGpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_gpu_data_address;
	}

	inline static u32 getDataSize(Tensor tensor)
	{
		return tensor.pTensorCore->mDataSize;
	}

	inline static u32 getDataSize(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->mDataSize;
	}

	inline static std::vector<u32> getTensorShape(const Tensor& tensor)
	{
		return  tensor.getShape();
	}
};