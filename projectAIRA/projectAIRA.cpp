#include "root_TensorNetwork.h"

int main()
{
	auto S = Split();
	auto seqAA = Sequential(Affine(10, 10), Affine(10, 10));
	Tensor t0{};

	auto t1 = seqAA(t0);
	auto u = S(t1);
}