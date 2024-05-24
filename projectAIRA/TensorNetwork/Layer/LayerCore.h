#pragma once
#include "Tensor/Tensor.h"


//�R���X�g���N�^�Ŏq�e���\����shared_ptr������this��o�^�������Čp���B
//��肪�N�����炱�����^���B
class LayerCore : public std::enable_shared_from_this<LayerCore>
{
public:
	using iotype = std::vector<Tensor>;

	LayerCore(u32 input_tensor_num, u32 output_tensor_num) 
		: m_input_tensor_num(input_tensor_num)
		, m_output_tensor_num(output_tensor_num)
		, mInputTensorCoreTbl(input_tensor_num)
		, m_child_tensorcore_tbl(output_tensor_num)
	{
		for (auto& child_tensorcore : m_child_tensorcore_tbl)
		{
			child_tensorcore = std::make_shared<TensorCore>();
		}
	}
	virtual ~LayerCore() {}

	iotype forwardCore(const iotype&);
	void regist_this_to_output_tensor()
	{
		//����������s�ǂ��N������������Ȃ��B
			//�Q�ƃG���[���o���炱�����^���B
		for (u32 i = 0; i < m_output_tensor_num; i++)
		{
			std::shared_ptr<LayerCore> shared_ptr_of_this = shared_from_this();
			m_child_tensorcore_tbl[i]->regist_parent_layercore(shared_ptr_of_this);
		}
	}
	void backward();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }

protected:
	virtual iotype forward(const iotype& input_tensors) = 0;

	//�p�����[�^�̃e�[�u��
	std::vector<TensorCore> mParameterTbl;

	//���̑w�����������e���\��
	//�i�e�w�̓e���\���p�̃������𒼐ڌ��Ă���C���[�W�j
	std::vector<std::shared_ptr<TensorCore>> m_child_tensorcore_tbl;

	//���`���ŃC���v�b�g���ꂽ�e���\�������o���Ă����p
	//���ꂪ�Ȃ��Ƌt�`���������ōs���Ȃ��Ȃ�B
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;

	//����Layer�̃��X�g
	std::vector<std::shared_ptr<LayerCore> > mInnerLayerCoreTbl;

	//���͂Əo�͂̃e���\���̐����L�^
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;
};