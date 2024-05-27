#include "Affine.h"
#include "Add.h"


AffineCore::AffineCore(u32 input_size, u32  output_size)
	: LayerCore(1, 1)
{
}

Layer Affine(u32 input_size, u32 output_size)
{
	return gen<AffineCore>("Affine", input_size, output_size);
}