#pragma once
#include "Layer.h"


class SplitCore : public LayerCore
{
public:
	SplitCore() : LayerCore(1, 2)
	{
		m_child_tensorcore_tbl.resize(2);
		m_child_tensorcore_tbl[0] = std::make_shared<TensorCore>();
		m_child_tensorcore_tbl[1] = std::make_shared<TensorCore>();
	}
	~SplitCore() {}

private:
	virtual iotype forward(const iotype& input_tensors)override
	{
		if (input_tensors.size() != 1)
		{
			std::cout << "Error : input tensor num of SplitCore is invalid ( " << input_tensors.size() << " )" << std::endl;
		}
		const Tensor& input_tensor = input_tensors[0];
		//for (u32 i = 0, end = input_tensor.getDataSize(); i< end; i++)
		{

		}

		return iotype{ Tensor(m_child_tensorcore_tbl[0]), Tensor(m_child_tensorcore_tbl[1]) };
	}
};


Layer Split()
{
	return gen<SplitCore>("Split");
}
