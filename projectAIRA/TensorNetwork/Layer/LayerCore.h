#pragma once
#include "Tensor/Tensor.h"


//コンストラクタで子テンソルにshared_ptr化したthisを登録したくて継承。
//問題が起きたらここを疑う。
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
		//ここが動作不良を起こすかもしれない。
			//参照エラーが出たらここを疑う。
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

	//パラメータのテーブル
	std::vector<TensorCore> mParameterTbl;

	//この層が生成したテンソル
	//（各層はテンソル用のメモリを直接見ているイメージ）
	std::vector<std::shared_ptr<TensorCore>> m_child_tensorcore_tbl;

	//順伝搬でインプットされたテンソル情報を覚えておく用
	//これがないと逆伝搬を自動で行えなくなる。
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;

	//内部Layerのリスト
	std::vector<std::shared_ptr<LayerCore> > mInnerLayerCoreTbl;

	//入力と出力のテンソルの数を記録
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;
};