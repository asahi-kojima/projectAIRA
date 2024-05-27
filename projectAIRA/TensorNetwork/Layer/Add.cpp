#include "Add.h"
#include "Tensor/TensorCore.h"


AddCore::AddCore()
	: LayerCore(2, 1, 1)
{
}

void AddCore::backward()
{
	std::cout << "Add backward" << std::endl;
}

Layer Add()
{
	Layer add_layer = gen<AddCore>("Add");
	//add_layer.mLayerCore->regist_this_to_output_tensor();
	return add_layer;
}
