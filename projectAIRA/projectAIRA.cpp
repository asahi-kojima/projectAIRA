#include "root_TensorNetwork.h"
#include <tuple>
#include <cassert>
std::tuple<Tensor, Tensor> convert(const LayerCore::iotype& tensor_vec)
{
	if (tensor_vec.size() != 2)
	{
		assert(0);
	}

	return std::tuple<Tensor, Tensor>(tensor_vec[0], tensor_vec[1]);
}




int main()
{
	auto affine = Affine(10, 10);
	auto seqAA = Sequential(Affine(10, 10), Affine(10, 10));
	auto S = Split();

	Tensor t0{};
	auto t1 = affine(t0);
	 t1 = seqAA(t1);
	auto [u0, u1] = convert(S(t1));
}