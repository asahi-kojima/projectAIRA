#include "root_TensorNetwork.h"
#include <tuple>
#include <cassert>
#include <Layer/Add.h>
#include "Layer/AddAsInner.h"
std::tuple<Tensor, Tensor> convert(const LayerCore::iotype& tensor_vec)
{
	if (tensor_vec.size() != 2)
	{
		assert(0);
	}

	return std::tuple<Tensor, Tensor>(tensor_vec[0], tensor_vec[1]);
}

void init(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getTensorDataSize(); i++)
	{
		t[i] = value;
	}
}

void init_linear(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getTensorDataSize(); i++)
	{
		t[i] = value * i;
	}
}

void confirm(const Tensor& tensor)
{
	const auto dataSize = Accessor2TensorCore::getDataSize(tensor);
	const auto pointer = Accessor2TensorCore::getAddressOnCpuFrom(tensor);

	std::vector<TensorCore::DataType> v(dataSize);
	for (u32 i = 0; i < dataSize; i++)
	{
		v[i] = pointer[i];
	}
	int x = 1 + 1;
}


int main()
{
	auto add0 = AddAsInner(); add0.mLayerName = "add0";
	auto add1 = Add(); add1.mLayerName = "add1";
	auto add2 = AddAsInner(); add2.mLayerName = "add2";

	Tensor t0(10, 3, 28, 28); init(t0, 1);
	Tensor s0(false, 10, 3, 28, 28); init_linear(s0, 2);
	for (u32 i = 0; i < 2; i++)
	{
		auto t1 = add0(t0, s0);
		t1[0].setName("t1");

		Tensor s0(10, 3, 28, 28); init(s0, 1);

		auto t2 = add1(t1[0], t1[0]);
		t2[0].setName("t2");


		Tensor s1(10, 3, 28, 28); init(s1, 1); s1.setName("s1");
		
		auto t3 = add2(t2[0], s1);
		t3[0].backward();
	}
}


//例えばこんなことをやりたい。
//class Affine : public LayerCore
//{
//public:
//	Affine() : LayerCore(param = 2)
//	{
//		mWeight = mParameterTbl[0];
//		mBias = mParameterTbl[1];
//
//		mInnerLayerCoreTbl.
//	}
//
//	Otype forward(auto input)
//	{
//
//	}
//
//
//private:
//	std::shared_ptr<TensorCore> mWeight;
//	std::shared_ptr<TensorCore> mBias;
//};
//
//class DNet : public LayerCore
//{
//public:
//	DNet() : LayerCore(param = 2)
//	{
//		mMainLayer = Sequential(Affine(), ReLU(), Affine(), ReLU());
//	}
//
//	iotype forward(const iotype& input) override
//	{
//		auto output = mMainLayer(input);
//		return output;
//	}
//
//private:
//	std::shared_ptr<LayerCore> mMainLayer;
//};
//
//class G : public LayerCore {};
//
//int exMain()
//{
//	auto d = DNet();
//	auto g = GNet();
//	auto lossNet = L1Loss();
//	auto optimizer = Adam();
//	for (u32 i = 0; i < 1000; i++)
//	{
//		auto z = Tensor{};
//		auto fake_img = g(z);
//		auto out = d(fake_img);
//		auto loss = lossNet(out, torch.ones());
//
//		d.zero_grad();
//		g.zero_grad();
//
//		loss.backward();
//		optimizer(d, g);
//
//
//	}
//}

