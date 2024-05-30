#pragma once
#include "Layer.h"
#include "Add.h"

class AddAsInnerCore : public LayerCore
{
public:
	AddAsInnerCore() : LayerCore(2,1)
	{
		mAdd = Add();
	}

	virtual iotype forward(const iotype& input_tensors) override
	{
		return mAdd(input_tensors);
	}

private:
	Layer mAdd;
};



inline Layer AddAsInner()
{
	return gen<AddAsInnerCore>("AddAsInner");
}