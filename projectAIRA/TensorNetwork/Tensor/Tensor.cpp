#include "Tensor.h"
#include "Layer/Add.h"

Tensor Tensor::operator+(const Tensor& t0)
{
	Layer add = Add();
	return add(*this, t0)[0];
}