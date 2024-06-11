#include "Optimizer.h"

using Optimizer = aoba::nn::optimizer::Optimizer;


Optimizer::Optimizer(u32 learningRate)
	:mLearningRate(learningRate)
	, mLinkedParameters(nullptr)
{
}

Optimizer::~Optimizer()
{
	LinkedList* current = mLinkedParameters;
	while (true)
	{
		auto next = current->next;
		delete current;

		if (!next)
		{
			break;
		}
	}
}

void Optimizer::operator()(const Layer& layer)
{
	const auto& layercore = *layer.getLayerCore();
	LinkedList* current = mLinkedParameters;
	while (true)
	{
		auto next = current->next;
		if (next)
		{
			current = next;
		}
		else
		{
			break;
		}
	}
	for (auto iter = layercore.m_parameter_tbl.begin(), end = layercore.m_parameter_tbl.end(); iter != end; iter++)
	{
		LinkedList* newtTable = new LinkedList(*iter, nullptr);
		current->next = newtTable;
		current = newtTable->next;
	}


	for (auto iter = layercore.m_internal_layer_tbl.begin(), end = layercore.m_internal_layer_tbl.end(); iter != end; iter++)
	{
		//(*this)(iter->second);
	}
}

void Optimizer::optimize()
{
	LinkedList* old = nullptr;
	LinkedList* current = mLinkedParameters;
	while (true)
	{
		if (current)
		{
			if (const std::shared_ptr<TensorCore>& tensorcore_as_shared = current->parameter.lock())
			{
				optimize_unique();

				old = current;
				current = current->next;
			}
			else
			{
				std::cout << "Resource unlock" << std::endl;
				if (old)
				{
					old->next = current->next;
					auto tmp = current;
					current = current->next;
					delete tmp;
				}
			}


		}
		else
		{
			break;
		}
	}
}
