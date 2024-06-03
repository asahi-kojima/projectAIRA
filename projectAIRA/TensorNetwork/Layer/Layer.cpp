#include "Add.h"
#include "Layer.h"

LayerCore::LayerCore(u32 input_tensor_num, u32 output_tensor_num)
	: m_input_tensor_num(input_tensor_num)
	, m_output_tensor_num(output_tensor_num)
	, mInputTensorCoreTbl(input_tensor_num)
	, m_parameter_tbl(0)
	, m_child_tensorcore_tbl(0)
	, mlayer(m_internal_layer_tbl)
	//, m_downstream_tensor_backward_finish(0)
{
}

LayerCore::LayerCore(u32 input_tensor_num, u32 output_tensor_num, u32 child_tensorcore_num)
	: m_input_tensor_num(input_tensor_num)
	, m_output_tensor_num(output_tensor_num)
	, mInputTensorCoreTbl(input_tensor_num)
	, m_parameter_tbl(0)
	, m_child_tensorcore_tbl(child_tensorcore_num), mlayer(m_internal_layer_tbl)
	//, m_downstream_tensor_backward_finish(child_tensorcore_num)
{
}

LayerCore::LayerCore(u32 input_tensor_num, u32 output_tensor_num, u32 child_tensorcore_num, u32 parameter_num)
	: m_input_tensor_num(input_tensor_num)
	, m_output_tensor_num(output_tensor_num)
	, mInputTensorCoreTbl(input_tensor_num)
	, m_parameter_tbl(parameter_num)
	, m_child_tensorcore_tbl(child_tensorcore_num), mlayer(m_internal_layer_tbl)
	//, m_downstream_tensor_backward_finish(child_tensorcore_num)
{
}


LayerCore::iotype LayerCore::callForward(const iotype& input_tensors)
{
	//���ʍ�Ƃ������ōs���B

	//���̃��C���[�̑g�ݍ��킹�ł͂Ȃ��w�ɑ΂��ẮA�����̏������s���B
	//���̃��C���[�ɓ��͂�C����悤�ȑw�ł������s���̂͏d�������ɂȂ�̂ŁA
	//���������B
	if (unique_implimention_layer)
	{
		//�܂��������ɗ^����ꂽ���̓e���\�������w�����߂��l�Ɉ�v���Ă��邩�m�F�B
		if (input_tensors.size() != m_input_tensor_num)
		{
			std::cout << "The number of input tensor must be " << m_input_tensor_num
				<< ". \nBut current input num is " << input_tensors.size() << "."
				<< std::endl;
			exit(1);
		}

		//���̓e���\���Ԃ�GPU���p�ݒ�ɖ������Ȃ����`�F�b�N����B
		bool on_cuda = input_tensors[0].pTensorCore->_m_on_cuda;
		for (u32 i = 1; i < m_input_tensor_num; i++)
		{
			if (input_tensors[i].pTensorCore->_m_on_cuda != on_cuda)
			{
				std::cout << "Between input tensor's, CPU/GPU setting contradict!" << std::endl;
				exit(1);
			}
		}

		if (m_init_finish && (on_cuda != m_on_cuda))
		{
			std::cout << "Between input and layer, CPU/GPU setting contradict!" << std::endl;
			exit(1);
		}

		//���͂��ꂽ�e���\�������̑w�̓��̓e���\���e�[�u���ɓo�^����B
		for (u32 i = 0; i < m_input_tensor_num; i++)
		{
			//�ߋ��ɓ��͂��������ꍇ�Ai�Ԗڂ̓��̓X���b�g�ɉߋ��̓��̓e���\�����o�^����Ă���B
			//����������ň�x��������B
			if (std::shared_ptr<TensorCore> p = mInputTensorCoreTbl[i].lock())
			{
				//�㗬�e���\���Ɉ˗����āA�o�����Ƀ����N��؂��Ă��炤�B
				p->disconnect_bidirection();
			}

			auto& tensorcore = input_tensors[i].pTensorCore;

			//�ߋ��ɂǂ����̑w�ɓ��͂���Ă����ꍇ�A�����̑w��񂪓o�^����Ă���B
			//�����ł������������B
			if (tensorcore->_m_downstream_layer)
			{
				tensorcore->disconnect_bidirection();
			}
			mInputTensorCoreTbl[i] = tensorcore;
			tensorcore->connect(shared_from_this(), i);
		}

		//���������ς�ł��Ȃ��ꍇ�A������ĂԂƃ��\�[�X�G���[�ɂȂ�B
		//�������A���������ς�łȂ��ꍇ�ɂ͈ȉ���forward�����Ń��\�[�X���m�ۂ���A������backward_finish��
		//�t���O��false�ɐݒ肳���̂ŁA���W�b�N�I�ɂ͂�����false�ɂ��Ă���̂Ɠ����ɂȂ��Ă���B
		if (m_init_finish)
		{
			for (const auto& child_tensorcore : m_child_tensorcore_tbl)
			{
				child_tensorcore->backward_finish = false;
			}
		}

	}
	//�e�w���̏��`�������ۂɎ��s����B
	return forward(input_tensors);
}

void LayerCore::callBackward()
{
	std::cout << "call backward\n";
	//�t�`���̏���
	for (const auto& child_tensorcore : m_child_tensorcore_tbl)
	{
		if (!child_tensorcore->backward_finish)
		{
			return;
		}
	}

	backward();


	//�㗬�֋t�`���̎w���𑗂�
	//��x���̓e���\���i���e�e���\���j�Ɏw���𑗂�A�����𒇉��
	//�㗬�w�ւƎw���𑗂�B
	for (const auto& input_tensor_core : mInputTensorCoreTbl)
	{
		if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
		{
			input_tensor_core_as_shared->backward_finish = true;

			//���z��񂪂���Ȃ��w�̍X�ɏ㗬�w�����z���͂���Ȃ��͂��Ȃ̂ŃX�L�b�v
			if (!input_tensor_core_as_shared->_m_need_grad)
			{
				continue;
			}

			input_tensor_core_as_shared->callBackward();
		}
		else
		{
			std::cout << "Resource error@LayerCore::callBackward" << std::endl;
			exit(1);
		}
	}
}

void LayerCore::regist_this_to_output_tensor()
{
	//����������s�ǂ��N������������Ȃ��B
		//�Q�ƃG���[���o���炱�����^���B
	for (u32 i = 0; i < m_output_tensor_num; i++)
	{
		std::shared_ptr<LayerCore> shared_ptr_of_this = shared_from_this();
		m_child_tensorcore_tbl[i]->regist_parent_layercore(shared_ptr_of_this);
	}
}



LayerCore::iotype operator+(const LayerCore::iotype& input0, const LayerCore::iotype& input1)
{
	Layer add = Add();
	return add(input0[0], input1[0]);
}


LayerCore::iotype operator+(const Tensor& input0, const Tensor& input1)
{
	Layer add = Add();
	return add(LayerCore::iotype{ input0, input1 });
}


//LayerCore::Layer::Layer(const Layer& layer)
//	:mLayerCore(layer.getLayerCore())
//	,mLayerName(layer.mLayerName)
//{}

LayerCore::Layer::Layer(const Layer& layer)
	:mLayerCore(layer.mLayerCore)
	, mLayerName(layer.mLayerName)
{}

LayerCore::Layer::Layer(const std::shared_ptr<LayerCore>& tensorcore, std::string name)
	:mLayerCore(tensorcore)
	, mLayerName(name)
{}

LayerCore::iotype Layer::operator()(const LayerCore::iotype& input) const
{
	return mLayerCore->callForward(input);
}