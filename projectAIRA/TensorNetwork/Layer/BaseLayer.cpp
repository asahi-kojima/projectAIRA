#include "BaseLayer.h"
#include "Tensor/Tensor.h"
#include "Layer.h"

namespace aoba::nn::layer
{

	BaseLayer::BaseLayer(u32 input_tensor_num, u32 output_tensor_num)
		: BaseLayer(input_tensor_num, output_tensor_num, 0, 0)
	{
	}

	BaseLayer::BaseLayer(u32 input_tensor_num, u32 output_tensor_num, u32 output_tensorcore_num)
		: BaseLayer(input_tensor_num, output_tensor_num, output_tensorcore_num , 0)
	{
	}

	BaseLayer::BaseLayer(u32 input_tensor_num, u32 output_tensor_num, u32 output_tensorcore_num, u32 trainable_parameter_num)
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

	const std::shared_ptr<tensor::TensorCore>& BaseLayer::getTensorCoreFrom(const Tensor& tensor)
	{
		return tensor.pTensorCore;
	}

	BaseLayer::iotype BaseLayer::callForward(const iotype& input_tensors)
	{
		//共通作業をここで行う。

		//他のレイヤーの組み合わせではない層に対しては、ここの処理を行う。
		//他のレイヤーに入力を任せるような層でここを行うのは重複処理になるので、
		//判定を入れる。
		if (having_unique_implimention)
		{
			//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
			if (input_tensors.size() != m_input_tensor_num)
			{
				std::cout << "The number of input tensor must be " << m_input_tensor_num << "." << std::endl;
				std::cout<< "But current input number is " << input_tensors.size() << "." << std::endl;
				exit(1);
			}

			//入力テンソル間のGPU利用設定に矛盾がないかチェックする。
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

			//入力されたテンソルをこの層の入力テンソルテーブルに登録する。
			for (u32 i = 0; i < m_input_tensor_num; i++)
			{
				//過去に入力があった場合、i番目の入力スロットに過去の入力テンソルが登録されている。
				//それをここで一度解除する。
				if (std::shared_ptr<TensorCore> p = mInputTensorCoreTbl[i].lock())
				{
					//上流テンソルに依頼して、双方向にリンクを切ってもらう。
					p->disconnect_bidirection();
				}

				auto& tensorcore = input_tensors[i].pTensorCore;

				//過去にどこかの層に入力されていた場合、下流の層情報が登録されている。
				//ここでそれを解除する。
				if (tensorcore->_m_downstream_layer)
				{
					tensorcore->disconnect_bidirection();
				}


				mInputTensorCoreTbl[i] = tensorcore;
				tensorcore->connect(shared_from_this(), i);
			}

			//下流テンソルの逆伝搬の進捗状況チェック表を初期化
			for (u32 i = 0; i < m_output_tensor_num; i++)
			{
				m_downstream_backward_checkTbl[i] = false;
			}
		}

		//各層毎の順伝搬を実際に実行する。
		return forward(input_tensors);
	}

	void BaseLayer::callBackward(u32 downstream_index)
	{
		//下流テンソル全ての逆伝搬が完了しているか確認
		{
			if (downstream_index >= m_output_tensor_num)
			{
				assert(0);
			}

			m_downstream_backward_checkTbl[downstream_index] = true;

			//逆伝搬の完了をループでチェック
			for (bool backward_finish : m_downstream_backward_checkTbl)
			{
				if (!backward_finish)
				{
					return;
				}
			}

			//全ての下流テンソルで逆伝搬が終了していれば逆伝搬を実施
			backward();
		}



		//上流へ逆伝搬の指示を送る
		//一度入力テンソル（＝親テンソル）に指示を送り、そこを仲介して
		//上流層へと指示を送る。
		for (const auto& input_tensor_core : mInputTensorCoreTbl)
		{
			if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
			{
				//勾配情報がないテンソルの上流層以上も勾配情報はいらないはずなのでスキップ
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

	void BaseLayer::initialize()
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

	void BaseLayer::genDownStreamTensor(u32 childNo)
	{
		if (childNo >= m_output_tensor_num)
		{
			assert(0);
		}

		//m_output_tensorcore_tbl[childNo] = tensorcore;
		m_output_tensorcore_tbl[childNo]->_m_location_in_upstream_layer = childNo;
		m_output_tensorcore_tbl[childNo]->regist_upstream_layer(shared_from_this());
	}

	Layer::Layer(const Layer& layer)
		:mBaseLayer(layer.mBaseLayer)
		, mLayerName(layer.mLayerName)
	{}

	Layer::Layer(const std::shared_ptr<BaseLayer>& tensorcore, std::string name)
		:mBaseLayer(tensorcore)
		, mLayerName(name)
	{}

	BaseLayer::iotype Layer::operator()(const BaseLayer::iotype& input) const
	{
		return mBaseLayer->callForward(input);
	}

}