#include "LayerCore.h"

LayerCore::iotype LayerCore::forwardCore(const iotype& input_tensors)
{
	//���ʍ�Ƃ������ōs���B

	//�܂��������ɗ^����ꂽ���̓e���\�������w�����߂��l�Ɉ�v���Ă��邩�m�F�B
	if (input_tensors.size() != m_input_tensor_num)
	{
		std::cout << "The number of input tensor must be " << m_input_tensor_num
			<< ". \nBut current inputNo is " << input_tensors.size() << "."
			<< std::endl;
	}

	//���͂��ꂽ�e���\�������̑w�̓��̓e���\���e�[�u���ɓo�^����B
	for (u32 i = 0; i < m_input_tensor_num; i++)
	{
		mInputTensorCoreTbl[i] = input_tensors[i].pTensorCore;
	}


	//�e�w���̏��`�������ۂɎ��s����B
	return forward(input_tensors);
}

void LayerCore::backward()
{
	//�t�`���̏���


	//�㗬�֋t�`���̎w���𑗂�
	//��x���̓e���\���i���e�e���\���j�Ɏw���𑗂�A�����𒇉��
	//�㗬�w�ւƎw���𑗂�B
	for (const auto& input_tensor_core : mInputTensorCoreTbl)
	{
		if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
		{
			input_tensor_core_as_shared->backward();
		}
		else
		{
			std::cout << "Resource error" << std::endl;
			exit(1);
		}
	}
}