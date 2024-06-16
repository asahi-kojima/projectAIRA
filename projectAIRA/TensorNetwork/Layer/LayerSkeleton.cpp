#include "Layer.h"
#include "Tensor/Tensor.h"
#include "nnLayer.h"

namespace aoba::nn::layer
{
	using LayerSkeleton = Layer::LayerSkeleton;

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num)
		: m_input_tensor_num(input_tensor_num)
		, m_output_tensor_num(output_tensor_num)
		, mInputTensorCoreTbl(input_tensor_num)
		, m_parameter_tbl(0)
		, m_child_tensorcore_tbl(0)
		, mlayer(m_internal_layer_tbl)

		, m_downstream_backward_checkTbl(output_tensor_num)
	{
	}

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num, u32 child_tensorcore_num)
		: m_input_tensor_num(input_tensor_num)
		, m_output_tensor_num(output_tensor_num)
		, mInputTensorCoreTbl(input_tensor_num)
		, m_parameter_tbl(0)
		, m_child_tensorcore_tbl(child_tensorcore_num)
		, mlayer(m_internal_layer_tbl)

		, m_downstream_backward_checkTbl(output_tensor_num)
	{
	}

	LayerSkeleton::LayerSkeleton(u32 input_tensor_num, u32 output_tensor_num, u32 child_tensorcore_num, u32 parameter_num)
		: m_input_tensor_num(input_tensor_num)
		, m_output_tensor_num(output_tensor_num)
		, mInputTensorCoreTbl(input_tensor_num)
		, m_parameter_tbl(parameter_num)
		, m_child_tensorcore_tbl(child_tensorcore_num)
		, mlayer(m_internal_layer_tbl)

		, m_downstream_backward_checkTbl(output_tensor_num)
	{
	}

	const std::shared_ptr<tensor::TensorCore>& LayerSkeleton::getTensorCoreFrom(const Tensor& tensor)
	{
		return tensor.pTensorCore;
	}

	LayerSkeleton::iotype LayerSkeleton::callForward(const iotype& input_tensors)
	{
		//共通作業をここで行う。

		//他のレイヤーの組み合わせではない層に対しては、ここの処理を行う。
		//他のレイヤーに入力を任せるような層でここを行うのは重複処理になるので、
		//判定を入れる。
		if (unique_implimention_layer)
		{
			//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
			if (input_tensors.size() != m_input_tensor_num)
			{
				std::cout << "The number of input tensor must be " << m_input_tensor_num
					<< ". \nBut current input num is " << input_tensors.size() << "."
					<< std::endl;
				exit(1);
			}

			//入力テンソル間のGPU利用設定に矛盾がないかチェックする。
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

			
			for (u32 i = 0; i < m_output_tensor_num; i++)
			{
				m_downstream_backward_checkTbl[i] = false;
			}
		}
		//各層毎の順伝搬を実際に実行する。
		return forward(input_tensors);
	}

	void LayerSkeleton::callBackward(u32 downstream_index)
	{
		{
			if (downstream_index >= m_output_tensor_num)
			{
				assert(0);
			}

			m_downstream_backward_checkTbl[downstream_index] = true;

			//逆伝搬の処理
			for (bool backward_finish : m_downstream_backward_checkTbl)
			{
				if (!backward_finish)
				{
					return;
				}
			}

			backward();
		}



		//上流へ逆伝搬の指示を送る
		//一度入力テンソル（＝親テンソル）に指示を送り、そこを仲介して
		//上流層へと指示を送る。
		for (const auto& input_tensor_core : mInputTensorCoreTbl)
		{
			if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
			{
				input_tensor_core_as_shared->backward_finish = true;

				//勾配情報がいらない層の更に上流層も勾配情報はいらないはずなのでスキップ
				if (!input_tensor_core_as_shared->_m_need_grad)
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

	void LayerSkeleton::regist_this_to_output_tensor()
	{
		//ここが動作不良を起こすかもしれない。
			//参照エラーが出たらここを疑う。
		for (u32 i = 0; i < m_output_tensor_num; i++)
		{
			std::shared_ptr<LayerSkeleton> shared_ptr_of_this = shared_from_this();
			m_child_tensorcore_tbl[i]->regist_parent_layercore(shared_ptr_of_this);
		}
	}



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