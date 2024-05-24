#include "TensorCore.h"
#include "Layer/LayerCore.h"

void TensorCore::backward() const
{
	if (std::shared_ptr<LayerCore> parentLayerCore = _m_parent_layer.lock())
	{
		parentLayerCore->backward();
	}
	else
	{
		std::cout << "Resource error" << std::endl;
		exit(1);
	}
}