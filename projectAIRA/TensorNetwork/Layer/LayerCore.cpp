#include "LayerCore.h"


LayerCore::LayerCore(u32 input_tensor_num, u32 output_tensor_num)
	: m_input_tensor_num(input_tensor_num)
	, m_output_tensor_num(output_tensor_num)
	, mInputTensorCoreTbl(input_tensor_num)
	, m_child_tensorcore_tbl(0)
{
}

LayerCore::LayerCore(u32 input_tensor_num, u32 output_tensor_num, u32 child_tensorcore_num)
	: m_input_tensor_num(input_tensor_num)
	, m_output_tensor_num(output_tensor_num)
	, mInputTensorCoreTbl(input_tensor_num)
	, m_child_tensorcore_tbl(child_tensorcore_num)
{
}


LayerCore::iotype LayerCore::callForward(const iotype& input_tensors)
{
	//共通作業をここで行う。

	//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
	if (input_tensors.size() != m_input_tensor_num)
	{
		std::cout << "The number of input tensor must be " << m_input_tensor_num
			<< ". \nBut current input num is " << input_tensors.size() << "."
			<< std::endl;
		exit(1);
	}


	bool on_cuda = input_tensors[0].pTensorCore->_m_on_cuda;
	for (u32 i = 1; i < m_input_tensor_num; i++)
	{
		if (input_tensors[i].pTensorCore->_m_on_cuda != on_cuda)
		{
			std::cout << "Between input tensor's, CPU/GPU setting contradict!" << std::endl;
			exit(1);
		}
	}

	if (m_init_finish && (on_cuda != m_use_gpu))
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


	//各層毎の順伝搬を実際に実行する。
	return forward(input_tensors);
}

void LayerCore::callBackward()
{
	std::cout << "call backward\n";
	//逆伝搬の処理
	backward();


	//上流へ逆伝搬の指示を送る
	//一度入力テンソル（＝親テンソル）に指示を送り、そこを仲介して
	//上流層へと指示を送る。
	for (const auto& input_tensor_core : mInputTensorCoreTbl)
	{
		if (std::shared_ptr<TensorCore> input_tensor_core_as_shared = input_tensor_core.lock())
		{
			//勾配情報がいらない層の更に上流層も勾配情報はいらないはずなのでスキップ
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
	//ここが動作不良を起こすかもしれない。
		//参照エラーが出たらここを疑う。
	for (u32 i = 0; i < m_output_tensor_num; i++)
	{
		std::shared_ptr<LayerCore> shared_ptr_of_this = shared_from_this();
		m_child_tensorcore_tbl[i]->regist_parent_layercore(shared_ptr_of_this);
	}
}