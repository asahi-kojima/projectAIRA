#include "Layer.h"
#include "Tensor/Tensor.h"
#include "nnLayer.h"

namespace aoba::nn::layer
{
	using LayerSkeleton = Layer::LayerSkeleton;

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num)
		: LayerSkeleton(input_tensor_num, output_tensor_num, 0, 0)
	{
	}

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num, u32 output_tensorcore_num)
		: LayerSkeleton(input_tensor_num, output_tensor_num, output_tensorcore_num , 0)
	{
	}

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num, u32 output_tensorcore_num, u32 trainable_parameter_num)
		: m_input_tensor_num(input_tensor_num)
		, m_output_tensor_num(output_tensor_num)
		, m_trainable_parameter_num(trainable_parameter_num)
		, mInputTensorCoreTbl(input_tensor_num)
		, mTrainableParameterTbl(0)
		, m_output_tensorcore_tbl(0)
		, mlayer(m_internal_layer_tbl)

		, m_downstream_backward_checkTbl(output_tensor_num)
	{
		for (u32 i = 0; i < trainable_parameter_num; i++)
		{
			mTrainableParameterTbl.push_back(std::make_shared<TensorCore>(true));
		}
		
		for (u32 i = 0; i < output_tensorcore_num; i++)
		{
			m_output_tensorcore_tbl.push_back(std::make_shared<TensorCore>(true));
		}
	}

	const std::shared_ptr<tensor::TensorCore>& LayerSkeleton::getTensorCoreFrom(const Tensor& tensor)
	{
		return tensor.pTensorCore;
	}

	LayerSkeleton::iotype LayerSkeleton::callForward(const iotype& input_tensors)
	{
		//���ʍ�Ƃ������ōs���B

		//���̃��C���[�̑g�ݍ��킹�ł͂Ȃ��w�ɑ΂��ẮA�����̏������s���B
		//���̃��C���[�ɓ��͂�C����悤�ȑw�ł������s���̂͏d�������ɂȂ�̂ŁA
		//���������B
		if (having_unique_implimention)
		{
			//�܂��������ɗ^����ꂽ���̓e���\�������w�����߂��l�Ɉ�v���Ă��邩�m�F�B
			if (input_tensors.size() != m_input_tensor_num)
			{
				std::cout << "The number of input tensor must be " << m_input_tensor_num << "." << std::endl;
				std::cout<< "But current input number is " << input_tensors.size() << "." << std::endl;
				exit(1);
			}

			//���̓e���\���Ԃ�GPU���p�ݒ�ɖ������Ȃ����`�F�b�N����B
			bool on_cuda = input_tensors[0].pTensorCore->m_on_cuda;
			for (u32 i = 1; i < m_input_tensor_num; i++)
			{
				if (input_tensors[i].pTensorCore->m_on_cuda != on_cuda)
				{
					std::cout << "Between input tensor's, CPU/GPU setting is not consistent!" << std::endl;
					exit(1);
				}
			}
			m_on_cuda = on_cuda;

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

			//�����e���\���̋t�`���̐i���󋵃`�F�b�N�\��������
			for (u32 i = 0; i < m_output_tensor_num; i++)
			{
				m_downstream_backward_checkTbl[i] = false;
			}
		}

		//�e�w���̏��`�������ۂɎ��s����B
		return forward(input_tensors);
	}

	void LayerSkeleton::callBackward(u32 downstream_index)
	{
		//�����e���\���S�Ă̋t�`�����������Ă��邩�m�F
		{
			if (downstream_index >= m_output_tensor_num)
			{
				assert(0);
			}

			m_downstream_backward_checkTbl[downstream_index] = true;

			//�t�`���̊��������[�v�Ń`�F�b�N
			for (bool backward_finish : m_downstream_backward_checkTbl)
			{
				if (!backward_finish)
				{
					return;
				}
			}

			//�S�Ẳ����e���\���ŋt�`�����I�����Ă���΋t�`�������{
			backward();
		}



		//�㗬�֋t�`���̎w���𑗂�
		//��x���̓e���\���i���e�e���\���j�Ɏw���𑗂�A�����𒇉��
		//�㗬�w�ւƎw���𑗂�B
		for (const auto& input_tensor_core : mInputTensorCoreTbl)
		{
			if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
			{
				//���z��񂪂Ȃ��e���\���̏㗬�w�ȏ�����z���͂���Ȃ��͂��Ȃ̂ŃX�L�b�v
				if (!input_tensor_core_as_shared->m_grad_required)
				{
					continue;
				}

				input_tensor_core_as_shared->callBackward();
			}
			else
			{
				std::cout << "Resource error@LayerSkeleton::callBackward" << std::endl;
				exit(1);
			}
		}
	}

	void LayerSkeleton::initialize()
	{
		if (m_init_finish)
		{
			assert(0);
		}

		for (u32 i = 0; i < m_output_tensor_num; i++)
		{
			genDownStreamTensor(i);
		}

		m_init_finish = true;
	}

	void LayerSkeleton::genDownStreamTensor(u32 childNo)
	{
		if (childNo >= m_output_tensor_num)
		{
			assert(0);
		}

		//m_output_tensorcore_tbl[childNo] = tensorcore;
		m_output_tensorcore_tbl[childNo]->_m_location_in_upstream_layer = childNo;
		m_output_tensorcore_tbl[childNo]->regist_upstream_layer(shared_from_this());
	}

	//void LayerSkeleton::regist_this_to_output_tensor()
	//{
	//	//����������s�ǂ��N������������Ȃ��B
	//		//�Q�ƃG���[���o���炱�����^���B
	//	for (u32 i = 0; i < m_output_tensor_num; i++)
	//	{
	//		std::shared_ptr<LayerSkeleton> shared_ptr_of_this = shared_from_this();
	//		m_output_tensorcore_tbl[i]->regist_parent_layercore(shared_ptr_of_this);
	//	}
	//}



	//LayerSkeleton::iotype operator+(const LayerSkeleton::iotype& input0, const LayerSkeleton::iotype& input1)
	//{
	//	Layer add = Add();
	//	return add(input0[0], input1[0]);
	//}


	//LayerSkeleton::iotype operator+(const tensor::Tensor& input0, const  tensor::Tensor& input1)
	//{
	//	Layer add = Add();
	//	return add(LayerSkeleton::iotype{ input0, input1 });
	//}


	//LayerSkeleton::Layer::Layer(const Layer& layer)
	//	:mLayerCore(layer.getLayerCore())
	//	,mLayerName(layer.mLayerName)
	//{}

	Layer::nnLayer::nnLayer(const nnLayer& layer)
		:mLayerSkeleton(layer.mLayerSkeleton)
		, mLayerName(layer.mLayerName)
	{}

	Layer::nnLayer::nnLayer(const std::shared_ptr<LayerSkeleton>& tensorcore, std::string name)
		:mLayerSkeleton(tensorcore)
		, mLayerName(name)
	{}

	LayerSkeleton::iotype Layer::nnLayer::operator()(const LayerSkeleton::iotype& input) const
	{
		return mLayerSkeleton->callForward(input);
	}

}