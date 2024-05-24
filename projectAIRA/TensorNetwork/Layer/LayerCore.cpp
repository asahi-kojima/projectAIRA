#include "LayerCore.h"

LayerCore::iotype LayerCore::forwardCore(const iotype& input_tensors)
{
	//共通作業をここで行う。

	//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
	if (input_tensors.size() != m_input_tensor_num)
	{
		std::cout << "The number of input tensor must be " << m_input_tensor_num
			<< ". \nBut current inputNo is " << input_tensors.size() << "."
			<< std::endl;
	}

	//入力されたテンソルをこの層の入力テンソルテーブルに登録する。
	for (u32 i = 0; i < m_input_tensor_num; i++)
	{
		mInputTensorCoreTbl[i] = input_tensors[i].pTensorCore;
	}


	//各層毎の順伝搬を実際に実行する。
	return forward(input_tensors);
}

void LayerCore::backward()
{
	//逆伝搬の処理


	//上流へ逆伝搬の指示を送る
	//一度入力テンソル（＝親テンソル）に指示を送り、そこを仲介して
	//上流層へと指示を送る。
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