#pragma once
#include "Layer.h"


class SequentialCore : public LayerCore
{
public:
	template <typename ... Args>
	SequentialCore(Args ... args) : LayerCore(1, 1)
	{
		//可変長テンプレートなので、分解してinnerModuleに格納している。
		Layer module_tbl[] = { args... };
		const u32 inner_module_num = (sizeof(module_tbl) / sizeof(module_tbl[0]));
		mInnerLayerPtrTbl.resize(inner_module_num);

		for (u32 i = 0, end = inner_module_num; i < end; i++)
		{
			if (module_tbl[i].mModule->get_input_tensor_num() != 1 || module_tbl[i].mModule->get_output_tensor_num() != 1)
			{
				std::cout << "inner module of Sequential  must be 1 input(your : "
					<< module_tbl[i].mModule->get_input_tensor_num() << " ) and 1 output(your : "
					<< module_tbl[i].mModule->get_output_tensor_num() << " ). " << std::endl;
				exit(1);
			}
			mInnerLayerPtrTbl[i] = module_tbl[i].mModule;
		}
	}

	virtual iotype forward(const iotype& input) override
	{
		if (input.size() != 1)
		{
			std::cout << "input tensor num is not 1" << std::endl;
		}

		return iotype();
	}
};

template<typename ... Args>
Layer Sequential(Args ... args)
{
	return gen<SequentialCore>(args...);
}