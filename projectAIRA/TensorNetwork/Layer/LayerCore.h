#pragma once
#include "Tensor/Tensor.h"


class LayerCore
{
public:
	using iotype = std::vector<Tensor>;

	LayerCore(u32 input_num, u32 output_num) : m_input_tensor_num(input_num), m_output_tensor_num(output_num) {}
	virtual ~LayerCore() {}


	virtual iotype forward(const iotype& input_tensors) = 0;
	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }

protected:

	//�p�����[�^�̃e�[�u��
	std::vector<TensorCore> mParameterTbl;

	//���̃��W���[�������������e���\��
	std::vector<std::shared_ptr<TensorCore> > mTensorCoreTbl;
	std::vector<std::shared_ptr<LayerCore> > mInnerLayerPtrTbl;

	//���͂Əo�͂̃e���\���̐����L�^
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;
};