#include "root_TensorNetwork.h"

int main()
{
	auto seqAA = Sequential(Affine(10, 10), Affine(10, 10));
	auto S = Split();
	Tensor t0{};

	auto t1 = seqAA(t0);
	auto u = S(t1);
}