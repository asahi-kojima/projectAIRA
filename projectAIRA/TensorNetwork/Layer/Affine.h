#pragma once
#include "Layer.h"

namespace aoba { namespace nn { namespace layer { class AffineCore; } } }


class   aoba::nn::layer::AffineCore : public LayerCore
{
public:
	AffineCore(u32 output_size);
	~AffineCore();

private:
	virtual iotype forward(const iotype& input_tensors) override;
	virtual void backward() override;

	u32 m_batch_size;
	u32 m_input_size;
	u32 m_output_size;
	const DataType affineWeight = 10.0f;
	void affine_forward_cpu_impl(const LayerCore::iotype&);
	void affine_backward_cpu_impl_input(const std::shared_ptr<TensorCore>&);
	void affine_backward_cpu_impl_parameter(const std::shared_ptr<TensorCore>&);
};


Layer Affine(u32 output_size);
