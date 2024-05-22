#pragma once
#include "Tensor/Tensor.h"


class LayerCore
{
public:
	using iotype = std::vector<Tensor>;

	LayerCore(u32 input_num, u32 output_num) : m_input_num(input_num), m_output_num(output_num) {}
	virtual ~LayerCore() {}


	virtual iotype forward(const iotype& input_tensors) = 0;
	u32 get_input_num() const { return m_input_num; }
	u32 get_output_num() const { return m_output_num; }

protected:

	//パラメータのテーブル
	std::vector<TensorCore> mParameterTbl;

	//このモジュールが生成したテンソル
	std::vector<std::shared_ptr<TensorCore> > mTensorCoreTbl;
	std::vector<std::shared_ptr<LayerCore> > mInnerModulePtrTbl;
	const u32 m_input_num;
	const u32 m_output_num;
};