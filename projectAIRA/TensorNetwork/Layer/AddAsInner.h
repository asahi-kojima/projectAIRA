#pragma once
#include "Layer.h"
#include "Add.h"

namespace aoba { namespace nn { namespace layer { class AddAsInnerCore; } } }


class aoba::nn::layer::AddAsInnerCore : public LayerCore
{
public:
	AddAsInnerCore() : LayerCore(2, 1)
	{
		mlayer["add"] = Add();
		mAdd = Add();
	}

	virtual iotype forward(const iotype& input_tensors) override
	{
		return mlayer["add"](input_tensors);
	}

private:
	Layer mAdd;
};



inline Layer AddAsInner()
{
	return aoba::nn::layer::gen<aoba::nn::layer::AddAsInnerCore>("AddAsInner");
}

